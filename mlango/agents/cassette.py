"""Recording an agent's model calls, and replaying them.

An eval suite with an agent in it is a suite that talks to a model: slow, paid
for by the call, and different every time. None of those are properties you want
in a test. The fix is older than agents — record the conversation once, replay it
afterwards — and it needs nothing from the agent, because every model call goes
through one method on one object.

So a cassette wraps a :class:`~mlango.agents.providers.Provider` rather than
reaching into the loop. Record against a real provider, commit the file, and the
same run happens offline and identically for everyone afterwards.

    from mlango.agents import RecordingProvider, ReplayProvider

    SupportAgent().run(
        "refund please",
        provider=RecordingProvider(get_provider("anthropic"), "cassettes/refund.json"),
    )

    SupportAgent().run("refund please", provider=ReplayProvider("cassettes/refund.json"))

The format is mlango's own, deliberately. glia records the same idea and sharing
its file would have been tidy, but its shape has nowhere to put
``Completion.raw_content`` — the assistant turn exactly as the provider wants it
echoed back. Dropping that would produce replays that pass against the cassette
and fail against the real API, which is the one thing a recording must never do.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from mlango.agents.providers.base import Completion, Provider, ToolCall, Usage
from mlango.core.exceptions import MlangoError

#: Bumped when a recorded file stops being readable by this code. Old cassettes
#: are then refused by name rather than misread.
FORMAT_VERSION = 1


class CassetteError(MlangoError):
    """A cassette could not be read, or did not contain the call being made."""


def _plain(value: Any) -> Any:
    """Whatever a provider handed back, as something JSON can hold.

    Provider-native content is usually a list of SDK objects. Every API that
    hands them over also accepts them back as plain dictionaries, so recording
    through JSON normalises them into the shape the next request wants anyway.
    """
    for attribute in ("model_dump", "dict", "to_dict"):
        method = getattr(value, attribute, None)
        if callable(method):
            try:
                return _plain(method())
            except TypeError:
                continue
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def request_key(payload: dict[str, Any]) -> str:
    """Content address for one call.

    Sorted keys, so a request assembled in a different order is still the same
    request. That is what lets a replay survive a refactor of the code that
    builds it.
    """
    canonical = json.dumps(_plain(payload), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _completion_to_dict(completion: Completion) -> dict[str, Any]:
    return {
        "text": completion.text,
        "tool_calls": [call.describe() for call in completion.tool_calls],
        "stop_reason": completion.stop_reason,
        "usage": completion.usage.describe(),
        "raw_content": _plain(completion.raw_content),
        "model": completion.model,
        "refusal": _plain(completion.refusal),
    }


def _usage_from_dict(data: dict[str, Any] | None) -> Usage:
    """Only the fields Usage is constructed from.

    ``describe()`` also reports ``total_tokens``, which is derived — handing it
    back to the constructor is a TypeError, and a recording that cannot be read
    is worse than none.
    """
    fields = {f.name for f in dataclasses.fields(Usage)}
    return Usage(**{k: int(v) for k, v in (data or {}).items() if k in fields})


def _completion_from_dict(data: dict[str, Any]) -> Completion:
    return Completion(
        text=data.get("text", ""),
        tool_calls=[
            ToolCall(
                id=call.get("id", ""),
                name=call.get("name", ""),
                arguments=call.get("arguments") or {},
            )
            for call in data.get("tool_calls", [])
        ],
        stop_reason=data.get("stop_reason", Completion.END_TURN),
        usage=_usage_from_dict(data.get("usage")),
        raw_content=data.get("raw_content"),
        model=data.get("model", ""),
        refusal=data.get("refusal"),
    )


class Cassette:
    """Recorded calls, as JSON on disk."""

    def __init__(self, interactions: list[dict[str, Any]] | None = None) -> None:
        self.interactions: list[dict[str, Any]] = list(interactions or [])

    def __len__(self) -> int:
        return len(self.interactions)

    def to_dict(self) -> dict[str, Any]:
        return {"version": FORMAT_VERSION, "interactions": self.interactions}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cassette:
        version = data.get("version")
        if version != FORMAT_VERSION:
            raise CassetteError(
                f"This cassette is format version {version!r}; this mlango reads "
                f"version {FORMAT_VERSION}. Re-record it against a live provider."
            )
        return cls(interactions=list(data.get("interactions", [])))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> Cassette:
        target = Path(path)
        if not target.is_file():
            raise CassetteError(
                f"No cassette at {target}. Record one first by running the agent with "
                f"RecordingProvider(...) wrapped around a live provider."
            )
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CassetteError(f"{target} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    def append(self, key: str, payload: dict[str, Any], completion: Completion) -> None:
        self.interactions.append(
            {"key": key, "request": _plain(payload), "response": _completion_to_dict(completion)}
        )


class RecordingProvider(Provider):
    """Passes every call to a real provider and writes down what came back.

    Saved after each call rather than at the end: a run that fails on its third
    step has still told you what the first two returned, which is usually the
    half you wanted.
    """

    name = "recording"

    def __init__(self, inner: Provider, path: str | Path, **options: Any) -> None:
        super().__init__(**options)
        self.inner = inner
        self.path = Path(path)
        self.cassette = Cassette()
        self.offline = inner.offline

    def complete(self, **kwargs: Any) -> Completion:
        completion = self.inner.complete(**kwargs)
        self.cassette.append(request_key(kwargs), kwargs, completion)
        self.cassette.save(self.path)
        return completion

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "recording": str(self.path), "inner": self.inner.name}


class ReplayProvider(Provider):
    """Answers from a cassette instead of from a model.

    Matched by content address first, so a run whose steps happen in a different
    order still replays. Otherwise the next unused recording in order, which is
    what lets a cassette survive an edit to the prompt that did not change what
    the model was being asked — unless ``strict``, where a call that was never
    recorded is an error rather than a guess.
    """

    name = "replay"
    offline = True

    def __init__(self, path: str | Path, *, strict: bool = False, **options: Any) -> None:
        super().__init__(**options)
        self.path = Path(path)
        self.cassette = Cassette.load(path)
        self.strict = strict
        self._used: set[int] = set()

    def complete(self, **kwargs: Any) -> Completion:
        key = request_key(kwargs)

        for index, interaction in enumerate(self.cassette.interactions):
            if index not in self._used and interaction.get("key") == key:
                self._used.add(index)
                return _completion_from_dict(interaction["response"])

        if self.strict:
            raise CassetteError(
                f"{self.path} has no recording for this call, and strict replay will not "
                f"guess. Re-record the cassette, or drop strict to fall back to order."
            )

        for index, interaction in enumerate(self.cassette.interactions):
            if index not in self._used:
                self._used.add(index)
                return _completion_from_dict(interaction["response"])

        raise CassetteError(
            f"{self.path} holds {len(self.cassette)} call(s) and the agent has now made "
            f"more than that. Re-record it against a live provider."
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "offline": True,
            "replaying": str(self.path),
            "calls": len(self.cassette),
            "strict": self.strict,
        }


__all__ = [
    "FORMAT_VERSION",
    "Cassette",
    "CassetteError",
    "RecordingProvider",
    "ReplayProvider",
    "request_key",
]

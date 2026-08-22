"""``_meta`` — the introspection object hanging off every declarative class.

Everything generic in mlango (the admin, migrations, the CLI, serialisation)
is written against ``_meta`` rather than against Dataset/Model/Agent
specifically. That is the single trick that lets one admin render four
different kinds of object.
"""

from __future__ import annotations

import json
from typing import Any

from mlango.core.exceptions import FieldError, ImproperlyConfigured
from mlango.core.fields import Field
from mlango.core.hashing import fingerprint

#: Meta options understood by every declarative class.
COMMON_META_OPTIONS = (
    "abstract",
    "app_label",
    "verbose_name",
    "verbose_name_plural",
    "description",
    "ordering",
    "tags",
)


class Options:
    def __init__(self, meta: type | None, owner_name: str, module: str, allowed: tuple[str, ...]):
        self.meta = meta
        self.object_name = owner_name
        self.module = module
        self.owner: type | None = None
        #: Which family this declaration belongs to: dataset, model, agent, eval.
        #: Set by the metaclass once the class hierarchy is known.
        self.kind: str = "object"

        self.abstract = False
        self.app_label: str | None = None
        self._verbose_name: str | None = None
        self._verbose_name_plural: str | None = None
        self.description: str = ""
        self.ordering: list[str] = []
        self.tags: list[str] = []

        self.local_fields: list[Field] = []
        self.fields: list[Field] = []
        self.extras: dict[str, Any] = {}

        self._allowed = tuple(dict.fromkeys(COMMON_META_OPTIONS + allowed))
        self._apply_meta()

    # -- construction --------------------------------------------------------

    def _apply_meta(self) -> None:
        if self.meta is None:
            return
        declared = {
            name: getattr(self.meta, name) for name in dir(self.meta) if not name.startswith("_")
        }
        unknown = sorted(set(declared) - set(self._allowed))
        if unknown:
            raise ImproperlyConfigured(
                f"{self.object_name}.Meta has unknown option(s): {', '.join(unknown)}. "
                f"Allowed here: {', '.join(sorted(self._allowed))}."
            )
        for name, value in declared.items():
            if name == "verbose_name":
                self._verbose_name = value
            elif name == "verbose_name_plural":
                self._verbose_name_plural = value
            elif name in COMMON_META_OPTIONS:
                setattr(self, name, value)
            else:
                self.extras[name] = value

    def inherit_extras(self, inherited: dict[str, Any]) -> None:
        """Fill in Meta options a base class declared and this one did not.

        ``abstract`` is deliberately excluded: inheriting it would make every
        subclass of an abstract base abstract too, so nothing could ever be
        declared concrete.
        """
        for key, value in inherited.items():
            if key == "abstract":
                continue
            self.extras.setdefault(key, value)

    def contribute_fields(self, fields: list[Field]) -> None:
        self.fields = sorted(fields, key=lambda f: f.creation_counter)

    def bind(self, owner: type) -> None:
        self.owner = owner
        if not self.description:
            doc = (owner.__doc__ or "").strip()
            self.description = doc.split("\n\n")[0].strip() if doc else ""

    # -- naming --------------------------------------------------------------

    @property
    def label(self) -> str:
        return f"{self.app_label}.{self.object_name}" if self.app_label else self.object_name

    @property
    def label_lower(self) -> str:
        return self.label.lower()

    @property
    def verbose_name(self) -> str:
        if self._verbose_name:
            return self._verbose_name
        # CamelCase -> "camel case"
        out: list[str] = []
        for index, char in enumerate(self.object_name):
            if char.isupper() and index and not self.object_name[index - 1].isupper():
                out.append(" ")
            out.append(char.lower())
        return "".join(out)

    @property
    def verbose_name_plural(self) -> str:
        if self._verbose_name_plural:
            return self._verbose_name_plural
        name = self.verbose_name
        # Dataset classes are usually named in the plural already (Reviews,
        # Sessions, Transactions), so a trailing "s" is treated as plural
        # rather than pluralised again into "Reviewses". Override
        # verbose_name_plural for the rare singular that ends in "s".
        if name.endswith("s"):
            return name
        if name.endswith(("x", "z", "ch", "sh")):
            return name + "es"
        if name.endswith("y") and not name.endswith(("ay", "ey", "oy", "uy")):
            return name[:-1] + "ies"
        return name + "s"

    # -- field access --------------------------------------------------------

    @property
    def field_names(self) -> list[str]:
        return [f.name or "" for f in self.fields]

    @property
    def fields_map(self) -> dict[str, Field]:
        return {f.name: f for f in self.fields if f.name}

    def get_field(self, name: str) -> Field:
        try:
            return self.fields_map[name]
        except KeyError as exc:
            available = ", ".join(self.field_names) or "(none)"
            raise FieldError(
                f"{self.label} has no field named {name!r}. Available: {available}."
            ) from exc

    def has_field(self, name: str) -> bool:
        return name in self.fields_map

    @property
    def target_fields(self) -> list[Field]:
        return [f for f in self.fields if f.is_target]

    @property
    def input_fields(self) -> list[Field]:
        return [f for f in self.fields if not f.is_target]

    @property
    def tunable_fields(self) -> list[Field]:
        return [f for f in self.fields if f.tunable]

    # -- serialisation -------------------------------------------------------

    def schema(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "object_name": self.object_name,
            "verbose_name": self.verbose_name,
            "description": self.description,
            "fields": [f.describe() for f in self.fields],
            "options": {k: v for k, v in sorted(self.extras.items()) if _simple(v)},
        }

    def recordable(self) -> dict[str, Any]:
        """The ``Meta`` options that survive a round trip through JSON.

        What a run can store about the thing it exercised, so a comparison
        between two runs can say *what changed* and not only that something
        did. Callables and live objects are excluded for the same reason
        migrations exclude them: they do not survive being written down.

        The test is an actual serialisation rather than a type check, because
        ``tools = [some_tool]`` is a list of live objects and a list is not
        made of strings by being a list. ``fingerprint`` keeps the shallower
        rule on purpose: tightening it would change every digest already
        written into a migration or a dataset version.
        """
        return {k: v for k, v in sorted(self.extras.items()) if _jsonable(v)}

    def fingerprint(self) -> str:
        """Stable hash of the declaration — the identity of a schema version."""
        return fingerprint(
            {
                "label": self.label,
                "fields": [f.describe() for f in self.fields],
                "options": self.recordable(),
            }
        )

    def __repr__(self) -> str:
        return f"<Options for {self.label}>"


def _jsonable(value: Any) -> bool:
    """Whether ``value`` really survives being written to a JSON column."""
    if not _simple(value):
        return False
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _simple(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict, tuple))

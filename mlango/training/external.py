"""Loading a model mlango did not train.

`diff` answers a question no registry answers for you: which rows the new
version lost. That answer does not depend on where the two models came from —
:func:`mlango.training.comparison.compare_predictors` needs two objects with a
``predict`` method and nothing else — but until now the only way to reach it was
to have trained both inside mlango.

So this is the door. Point it at two artefacts you already have, declare the
dataset they should be scored on, and the comparison is the same comparison.
Nothing here asks you to adopt the framework first, which is the point: a team
with a pickled model and a CSV can get an answer before deciding anything.

A URI is either a path, or ``scheme:rest`` where the scheme is registered under
the ``mlango.loaders`` entry-point group. Registries other than mlango's belong
in that group rather than in here — each one drags in a client library, and a
framework that installs somebody else's SDK to read a file is not one you want.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mlango.core.exceptions import MlangoError
from mlango.core.plugins import discover

#: Entry-point group for loaders that know a registry mlango does not.
LOADER_GROUP = "mlango.loaders"


@runtime_checkable
class Predictor(Protocol):
    """The whole interface a comparison needs."""

    def predict(self, inputs: Any) -> Any: ...


def load_predictor(uri: str) -> Predictor:
    """Load a model from ``uri`` and return something that can predict.

    Raises:
        MlangoError: when the scheme has no loader, the file is missing, or what
            came back cannot predict. Each message names the fix, because this
            is the first thing a new user runs and the first error they see.
    """
    scheme, _, rest = uri.partition(":")

    # A Windows path starts `C:\...`; a one-letter scheme is a drive.
    if not rest or len(scheme) == 1:
        loaded = _load_file(Path(uri))
    elif scheme == "file":
        loaded = _load_file(Path(rest))
    else:
        loaded = _load_via_plugin(scheme, rest, uri)

    if not hasattr(loaded, "predict"):
        raise MlangoError(
            f"{uri} loaded a {type(loaded).__name__}, which has no predict(). "
            f"A comparison needs two things that can be asked for predictions."
        )
    return loaded  # type: ignore[return-value]


def _load_file(path: Path) -> Any:
    if not path.is_file():
        raise MlangoError(
            f"No such file: {path}. Pass a path to a saved model, or "
            f"scheme:reference for a registry — see `installed_loaders()`."
        )

    # joblib first: it is what scikit-learn tells people to use, and it reads
    # plain pickles too. Falling back keeps the common case dependency-free.
    try:
        import joblib
    except ImportError:
        pass
    else:
        return joblib.load(path)

    import pickle

    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - the user named this file


def _load_via_plugin(scheme: str, rest: str, uri: str) -> Any:
    available = discover(LOADER_GROUP)
    target = available.get(scheme)
    if target is None:
        known = ", ".join(sorted(available)) or "none are installed"
        raise MlangoError(
            f"No loader for {scheme!r} in {uri}. Registered schemes: {known}. "
            f"A loader is a package exposing `{scheme} = module:function` under "
            f"the {LOADER_GROUP!r} entry-point group."
        )

    module_name, _, attribute = str(target).partition(":")
    from importlib import import_module

    loader = getattr(import_module(module_name), attribute)
    return loader(rest)


def installed_loaders() -> dict[str, str]:
    """Schemes `load_predictor` can resolve, beyond a plain path."""
    return dict(discover(LOADER_GROUP))


def columns_for(
    dataset_class: Any,
    *,
    features: list[str] | None = None,
    target: str | None = None,
) -> tuple[list[str], str]:
    """Work out what to feed the models and what to score them against.

    A declared Model answers this from its own ``Meta``. An artefact on disk has
    no Meta, so the dataset answers instead: its declared target, and every
    other field except the primary key — an id is bookkeeping, and scoring on it
    is the same leak it would be during training.
    """
    if target is None:
        targets = dataset_class._meta.target_fields
        if len(targets) != 1:
            raise MlangoError(
                f"{dataset_class._meta.label} declares {len(targets)} target fields, so "
                f"there is no single column to score against. Name one with --target."
            )
        target = str(targets[0].name or "")

    if features is None:
        # Mirrors Model.get_features: every declared field except the target and
        # the primary key.
        excluded = {target}
        primary = dataset_class._meta.extras.get("primary_key")
        if primary:
            excluded.add(str(primary))
        features = [
            str(field.name or "")
            for field in dataset_class._meta.fields
            if (field.name or "") not in excluded
        ]
    else:
        unknown = [name for name in features if not dataset_class._meta.has_field(name)]
        if unknown:
            raise MlangoError(
                f"{dataset_class._meta.label} has no field(s) {', '.join(unknown)}. "
                f"Declared: {', '.join(str(f.name) for f in dataset_class._meta.fields)}."
            )

    if not features:
        raise MlangoError(
            f"{dataset_class._meta.label} has no feature columns once {target!r} is "
            f"set aside. Name them with --features."
        )
    return features, target


__all__ = ["LOADER_GROUP", "Predictor", "columns_for", "installed_loaders", "load_predictor"]

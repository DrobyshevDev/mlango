"""The comparison from the README, reproducible in one command.

    python examples/promotion/promotion.py

Two honest models on noisy data. The second is more accurate overall *and*
gets eleven rows wrong that the first got right — which is the whole point:
no aggregate metric will tell you about those eleven.

The data is synthetic on purpose. Real label noise (12% of rows carry the
wrong label) is what keeps both models realistically imperfect; without it a
TF-IDF model memorises an invented vocabulary and every comparison is
degenerate at 1.0 accuracy.
"""

import os
import random
import sys
import tempfile

from mlango.conf import settings

settings.configure(
    BASE_DIR=tempfile.mkdtemp(prefix="mlango-readme-"),
    METASTORE={"URL": "sqlite:///readme.db"},
    STORAGE={"BACKEND": "mlango.storage.local.LocalStorage", "ROOT": "artifacts"},
    DEFAULT_PROVIDER="echo",
    INSTALLED_APPS=[],
    SEED=7,
)

import mlango  # noqa: E402

mlango.setup()

from mlango.core import fields  # noqa: E402
from mlango.data import Dataset, InMemorySource  # noqa: E402
from mlango.training import Model  # noqa: E402

POSITIVE = "delightful brilliant wonderful excellent warm moving clever sharp".split()
NEGATIVE = "dull dreadful tedious clumsy shallow weak muddled bland".split()
FILLER = "film movie story cast script scene ending pace".split()

rng = random.Random(11)


def review(label: str) -> str:
    """A short review with enough noise that no model gets it all right."""
    words = POSITIVE if label == "pos" else NEGATIVE
    body = [rng.choice(words) for _ in range(2)] + [rng.choice(FILLER) for _ in range(4)]
    rng.shuffle(body)
    return " ".join(body)


ROWS = []
for index in range(500):
    truth = "pos" if index % 2 else "neg"
    # Real label noise, not vocabulary noise: without it a TF-IDF model memorises
    # an invented vocabulary perfectly and every comparison is degenerate.
    label = truth if rng.random() > 0.12 else ("neg" if truth == "pos" else "pos")
    ROWS.append({"id": index, "text": review(truth), "label": label})


class Reviews(Dataset):
    """Customer product reviews."""

    id = fields.IntegerField()
    text = fields.TextField()
    label = fields.LabelField(["neg", "pos"])

    class Meta:
        app_label = "reviews"
        source = InMemorySource(ROWS)
        primary_key = "id"


class Sentiment(Model):
    """TF-IDF into logistic regression."""

    max_features = fields.IntegerField(default=20_000, tunable=True)
    C = fields.FloatField(default=1.0, min_value=0.0, tunable=True)

    class Meta:
        app_label = "reviews"
        dataset = Reviews
        trainer = "sklearn"
        task = "classification"
        features = ["text"]

    def build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            TfidfVectorizer(max_features=self.max_features),
            LogisticRegression(C=self.C, max_iter=1000),
        )


if __name__ == "__main__":
    older = Sentiment.fit(max_features=14, C=0.9)._version.version
    newer = Sentiment.fit(max_features=16, C=1.4)._version.version

    from mlango.management.manager import load_command

    command = load_command("diff", "mlango.management.commands.diff")
    sys.exit(command.run_from_argv([Sentiment._meta.label, str(older), str(newer)]))

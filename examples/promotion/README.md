# The promotion question

```bash
python examples/promotion/promotion.py
```

This is the comparison the [project README](../../README.md) opens with,
reproducible in one command. It needs `mlango[sklearn]` and nothing else — no
project, no configuration, no data to download.

```
reviews.Sentiment v1 → v2 on 500 rows of reviews.Reviews

  agreement      92.0%
  changed        40 row(s)
    pos → neg                22
    neg → pos                18

Against the labels
  v1     accuracy     0.7700
  v2     accuracy     0.8060   +0.0360
  fixed          29 row(s) wrong in v1
  broke          11 row(s) right in v1
  verdict        a real improvement: 29 fixed against 11 broken (p=0.006)
```

v2 is better by every summary you would normally look at. It is also wrong on
**eleven rows v1 got right**, and `broke` is the only place that number appears.
On a real dataset those eleven are usually the ones somebody complained about.

The verdict line is McNemar's test, computed exactly: 29 against 11 is unlikely
enough to be a coin (p=0.006) that the improvement is real. Twenty-nine against
twenty-two would not have been, and the report would have said so.

## Why the data is synthetic

Deliberately. Twelve per cent of the rows carry the wrong label, which is what
keeps both models realistically imperfect — without it a TF-IDF pipeline
memorises an invented vocabulary, both versions score 1.0, and the comparison
has nothing to show.

Everything else is real: real training runs, real registered versions, and the
real `manage.py diff` command reading them back out of the metastore.

## Next

```bash
python manage.py diff reviews.Sentiment 1 2 --show-changes 10   # which rows moved
python manage.py diff reviews.Sentiment 1 2 --fail-on-regression # exit code for CI
```

See [Command line](https://drobyshevdev.github.io/mlango/cli/) for the rest —
comparing evaluation runs, artefacts mlango never trained, and live traffic.

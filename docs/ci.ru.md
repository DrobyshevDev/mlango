# Непрерывная интеграция

Сравнение, прочитанное в терминале, — это решение, принятое одним человеком.
Промоут модели редко бывает решением одного человека, а остальные находятся в
пул-реквесте, — значит, сравнение должно уметь попадать туда: на машину без
терминала, где никто не смотрит.

Это делают два флага. `--fail-on-regression` превращает сравнение в код
возврата, а `--format markdown` — в текст, который человек прочитает там, где он
и так есть.

## Отчёт как комментарий

```bash
python manage.py diff reviews.Sentiment --format markdown --show-changes 20
```

```markdown
<!-- mlango:diff:model:reviews.Sentiment -->

### ⚠️ reviews.Sentiment v1 → v2

**11 rows broken**, 29 fixed, over 500 rows of `reviews.Reviews`.

| | v1 → v2 |
|---|---:|
| `accuracy` | 0.7700 → **0.8060** (+0.0360) |
| agreement | 92.0% |
| changed | 40 rows |
| fixed | 29 rows |
| broke | **11 rows** |

Movement: `pos → neg` 22 · `neg → pos` 18

> a real improvement: 29 fixed against 11 broken (p=0.006)

<details>
<summary>20 of 40 rows where they disagree</summary>
…
</details>
```

Число сломанных строк стоит в заголовке и в первом предложении, потому что
комментарий в пул-реквесте чаще всего читают как уведомление и не открывают.
Строки, наоборот, спрятаны: тому, кто **открыл**, нужны доказательства, а
комментарий в двести строк никто не пролистывает.

Рендерятся те же четыре сравнения, которые команда и так умеет делать, —
зарегистрированные версии, прогоны оценок, два файла, которые mlango не обучал, и
живой трафик из теневого деплоя, — потому что это функция от отчёта, а не от
команды.

| Флаг | Что делает |
|---|---|
| `--format text\|markdown\|json` | Как рендерить. По умолчанию `text` |
| `--output PATH` | Записать в файл. На код возврата не влияет |
| `--show-changes N` | Включить до N расходящихся строк |

`--json` — старое написание `--format json`, и оно по-прежнему значит то же
самое.

## Workflow для GitHub Actions

```yaml
name: model

on: pull_request

permissions:
  contents: read
  pull-requests: write

jobs:
  diff:
    runs-on: ubuntu-latest
    env:
      DATABASE_URL: ${{ secrets.DATABASE_URL }}
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"

      - run: pip install -r requirements.txt

      - name: Обучить кандидата
        run: python manage.py train reviews.Sentiment -v 0

      - name: Сравнить с тем, что в проде
        id: diff
        continue-on-error: true
        run: |
          python manage.py diff reviews.Sentiment \
            --format markdown --show-changes 20 \
            --output diff.md \
            --fail-on-regression significant

      - name: Сказать об этом в пул-реквесте
        if: always()
        env:
          GH_TOKEN: ${{ github.token }}
          NUMBER: ${{ github.event.number }}
        run: gh pr comment "$NUMBER" --body-file diff.md --edit-last --create-if-none

      - name: Упасть, если стало хуже
        if: steps.diff.outcome == 'failure'
        run: exit 1
```

Три вещи здесь сделаны намеренно.

**`continue-on-error`, а потом явное падение.** Сравнение должно быть
опубликовано независимо от того, прошло оно или нет: задача, которая краснеет, не
объясняя почему, — это ровно то, что здесь и заменяется. Поэтому diff'у
разрешено упасть, комментарий публикуется, а падение поднимается после.

**`--edit-last --create-if-none`** заменяет предыдущий комментарий, а не
добавляет новый, — и на ветке с девятью пушами будет один отчёт, а не тред,
который никто не читает.

**`significant`, а не голый флаг.** На реальных данных «строго лучшая» версия —
выдумка: что-нибудь всегда регрессирует. Голый флаг падает на одной потерянной
строке: это правильное правило для выверенного набора и неправильное здесь.
`significant` падает, только когда потери перевешивают приобретения сильнее, чем
это объясняется случайностью, — см.
[Сравнение двух версий](cli.md#diffing-two-versions).

### Несколько моделей в одной задаче

`--edit-last` находит последний комментарий от аккаунта самой задачи, а это уже
не тот комментарий, как только отчитывается вторая модель. Каждый отрендеренный
отчёт начинается с маркера, называющего то, что сравнивали:

```
<!-- mlango:diff:model:reviews.Sentiment -->
```

Он стабилен между запусками и уникален для модели, поэтому задача, сравнивающая
несколько, может найти комментарий каждого отчёта и обновить именно его.

## Что CI должен видеть

Сравнение читает реестр. В свежем чекауте реестра нет, поэтому задача, которая
обучает и сравнивает с `sqlite:///mlango.db`, сравнивает кандидата ни с чем.

Направьте раннер на тот метастор, где промоуты действительно происходят:

```python
METASTORE = {"URL": os.environ.get("DATABASE_URL", "sqlite:///mlango.db")}
```

Обычный ответ — Postgres. Если ваш реестр — файл SQLite, который где-то лежит,
подойдёт и шаг, скачивающий его перед сравнением: доступа на чтение достаточно,
потому что diff ничего не регистрирует.

## GitLab CI

Те же команды и тот же довод в пользу того, чтобы разделить падение и отчёт:

```yaml
model-diff:
  image: python:3.12
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  script:
    - pip install -r requirements.txt
    - python manage.py train reviews.Sentiment -v 0
    - |
      python manage.py diff reviews.Sentiment \
        --format markdown --show-changes 20 \
        --output diff.md --fail-on-regression significant
  artifacts:
    when: always
    paths:
      - diff.md
```

`when: always` — та же мысль, что и `continue-on-error`: отчёт нужен именно
тогда, когда задача упала.

## И для оценок тоже

У агента нет номера версии, поэтому сравнивать надо два прогона его набора
оценок. Всё сказанное выше работает без изменений:

```bash
python manage.py evaluate support.AnswerQuality
python manage.py diff --eval support.AnswerQuality \
    --format markdown --output diff.md --fail-on-regression significant
```

Это правка промпта, закрытая тем же барьером, что и промоут, — и отчёт назовёт,
что вы изменили (промпт, модель, лимит шагов), рядом с тем, к чему это привело.

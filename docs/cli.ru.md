# Командная строка

В каждом проекте есть `manage.py`. Скрипт `mlango` делает то же самое, когда
проекта ещё нет, а `python -m mlango` работает, если скрипта нет в `PATH`.

```bash
python manage.py help
python manage.py help train
```

## Команды

### Начало работы

| Команда | Что делает |
|---|---|
| `mlango startproject NAME [DIR]` | Создаёт проект, который уже работает. `--bare` пропускает демо-приложение |
| `manage.py startapp NAME` | Создаёт приложение: datasets, models, agents, evals, admin, migrations, tests |
| `manage.py check` | Проверяет настройки, бэкенды, связи, миграции и админку |
| `mlango startplugin NAME --kind trainer` | Создаёт публикуемый пакет, расширяющий mlango |

`startplugin` не нужен проект: он пишет дистрибутив — pyproject с уже
объявленным entry point, контракт с комментариями в интересных местах, LICENSE и
тесты, — так что проекту достаточно `pip install`. `--kind` — это `trainer`,
`provider`, `storage` или `source`. См. [Расширение](extending.md).

### Свои данные { #bringing-your-own-data }

В Django есть `inspectdb` для существующей базы. Здесь то же самое для файла:
команда читает выборку и печатает `Dataset`, который можно вставить в
`datasets.py` — так первое объявление становится правкой, а не пустым листом.

```bash
python manage.py inspectdata data/reviews.csv
python manage.py inspectdata data/reviews.csv --name Feedback -n 5000
python manage.py inspectdata data/reviews.csv --write --app reviews
```

Читает `.csv`, `.tsv`, `.jsonl`, `.ndjson`, `.json` и `.parquet`. Своих
объявлений ей не нужно, поэтому она работает на только что созданном проекте.

```python
class Reviews(Dataset):
    """40 rows, 6 columns."""

    id = IntegerField(min_value=1, max_value=40)
    body = TextField()
    stars = IntegerField(min_value=1, max_value=5)
    country = CharField(max_length=16, choices=["GB", "US"])
    verified = BooleanField()
    label = LabelField(["neg", "pos"])

    class Meta:
        source = CSVSource("data/reviews.csv")
        primary_key = "id"
```

Как она решает:

| Признак | Становится |
|---|---|
| Все значения — целые числа | `IntegerField` с наблюдённым диапазоном |
| Хоть одно значение с точкой | `FloatField` с наблюдённым диапазоном |
| `true`/`yes`/`t`/`on` и противоположные | `BooleanField` |
| dict, list или строка, разбираемая как они | `JSONField` |
| Метки времени в ISO | `DateTimeField` |
| Мало различных значений, и они повторяются | `CharField(choices=…)` |
| Хоть одно значение длиннее 32 символов | `TextField` |
| Колонка с именем `label`, `target`, `y`, `class`… | `LabelField` или `TargetField` |
| Уникальная колонка `id`, `uuid` или `*_id` | `Meta.primary_key` |
| Часть значений пуста | `null=True, required=False` |

Два правила, которые стоит знать. **Целевой становится ровно одна колонка** —
две оставили бы `Model.get_target()` без выбора, поэтому остальные категориальные
остаются `CharField` с `choices`. И `max_length` выставляется только когда все
значения в выборке короткие: слишком маленький предел позже отвергнет валидные
данные, а `TextField` не отвергает ничего.

Это отправная точка, а не истина. Всё, что она угадала, помечено комментарием, а
имя колонки, которое не может быть атрибутом Python, названо явно, а не молча
искажено.

### Данные

```bash
python manage.py dataset list
python manage.py dataset show reviews.Reviews
python manage.py dataset head reviews.Reviews -n 20
python manage.py dataset validate reviews.Reviews
python manage.py dataset materialize reviews.Reviews --notes "ночной снимок"
python manage.py dataset versions reviews.Reviews
```

### Миграции

```bash
python manage.py makemigrations [app] [-n NAME] [--dry-run] [--empty]
python manage.py migrate [app] [--plan] [--fake]
python manage.py showmigrations [app]
```

### Обучение

```bash
python manage.py train reviews.Sentiment -p C=2.0 -p max_features=5000 \
    --tag baseline --notes "первая попытка" --materialize

python manage.py sweep reviews.Sentiment -p C=0.25,1,4 \
    --strategy grid --metric accuracy --mode max --promote-best production
```

| Флаг | Что делает |
|---|---|
| `-p NAME=VALUE` | Переопределяет гиперпараметр. Можно повторять |
| `--dataset LABEL` | Обучает на другом датасете |
| `--tag TAG` | Помечает запуск тегом. Можно повторять |
| `--seed N` | Переопределяет seed |
| `--materialize` | Сначала фиксирует обучающую выборку как версию датасета |
| `--no-register` | Обучает, не добавляя в реестр версий |

### Предсказание { #prediction }

Оценка без запуска сервера. Модель берётся из реестра версий, то есть работает
тот же артефакт, который отдавал бы API.

```bash
python manage.py predict reviews.Sentiment "понравилось от начала до конца"
python manage.py predict reviews.Sentiment "отлично" "ужасно" --proba

python manage.py predict reviews.Sentiment --dataset -n 100
python manage.py predict reviews.Sentiment --dataset --filter label=pos

python manage.py predict reviews.Sentiment --file incoming.jsonl \
    --format jsonl --output scored.jsonl
```

| Флаг | Что делает |
|---|---|
| `--dataset` | Оценить объявленный датасет модели |
| `--filter FIELD=VALUE` | Сузить датасет. Можно повторять |
| `--file PATH` | Оценить файл csv/tsv/jsonl/json/parquet |
| `-n N` | Остановиться после N записей |
| `--version N` / `--stage NAME` | Какую версию из реестра загрузить |
| `--proba` | Добавить вероятности классов |
| `--format table\|jsonl\|csv` | Как выводить |
| `--output PATH` | Записать в файл вместо stdout |

Если во входных данных есть `id`, `uuid` или `pk`, он попадает в вывод — так
оценённый файл можно соединить с источником. Если в данных нет признака, который
нужен модели, команда назовёт отсутствующую колонку и перечислит имеющиеся,
вместо того чтобы уронить тренер где-то внутри векторизатора.

### Объяснение версии

На какие признаки на самом деле опиралась обученная версия. Веса записываются в
строку версии при регистрации, поэтому команда читает метастор и не загружает
артефакт:

```bash
python manage.py explain reviews.Sentiment
python manage.py explain reviews.Sentiment --stage production -n 10
python manage.py explain reviews.Sentiment --json
```

```
reviews.Sentiment@v4
top 10 of 40, largest weight first

delightful   ████████████████████████████████  2.4439
dull         ████████████████████████████···· -2.1614
brilliant    ███████████████████████████·····  2.0495
boring       ███████████████████████████····· -2.0407
badly        ██████████████████████████······ -1.9794
beautifully  ██████████████████████████······  1.9708
awful        █████████████████████████······· -1.8902
excellent    █████████████████████████·······  1.8844
waste        ███████████████████············· -1.4853
every        ███████████████████·············  1.4288
```

Векторайзер в пайплайне называет свои колонки сам — именно это превращает 40 000
безымянных ячеек в слова выше. Знак — это направление эффекта: он сохраняется для
бинарных и регрессионных фитов, где что-то значит, и отбрасывается для
многоклассовых, где признак «за» один класс одновременно «против» другого.

| Флаг | Что делает |
|---|---|
| `--version N` / `--stage NAME` | Какую версию объяснять (по умолчанию — последнюю) |
| `-n N` | Сколько признаков показать |
| `--json` | Отдать веса вместо диаграммы |
| `--recompute` | Загрузить артефакт, пересчитать веса и сохранить их |

`--recompute` — запасной выход для версии, зарегистрированной до того, как mlango
научился объяснять. Бэкенды, которые не могут назвать признак — нейросетевые, —
не сообщают ничего, вместо того чтобы выдумать правдоподобный список.

### Сравнение двух версий

Агрегированные метрики отвечают на вопрос «новая лучше?» и прячут тот ответ,
которого вы боитесь: версия, которая на два пункта точнее в среднем, могла
сломать сорок строк, работавших раньше. Эта команда прогоняет обе по одним и тем
же данным и сравнивает ответы.

```bash
python manage.py diff reviews.Sentiment 3 4
python manage.py diff reviews.Sentiment                      # production против последней
python manage.py diff reviews.Sentiment 3 4 --show-changes 20
python manage.py diff reviews.Sentiment 3 4 --fail-on-regression
```

```
reviews.Sentiment v3 → v4 on 500 rows of reviews.Reviews

  agreement      94.2%
  changed        29 row(s)
    neg → pos                18
    pos → neg                11

Against the labels
  v3 accuracy      0.8840
  v4 accuracy      0.9020   +0.0180
  fixed          22 row(s) wrong in v3
  broke           4 row(s) right in v3
  verdict        a real improvement: 22 fixed against 4 broken (p=0.001)
```

**`broke`** — то число, которое никто не показывает и которое всем нужно. Промоут,
поднимающий среднее ценой строк, работавших раньше, — это ровно тот промоут,
который откатывают через неделю; `--fail-on-regression` превращает его в код
возврата, который можно поставить перед промоутом.

**`verdict`** отвечает на вопрос, который эти два числа сами и порождают. Строки,
где обе версии правы, ничего не говорят о том, какая лучше; строки, где обе
неправы, — тоже. Информацию несут только расхождения. Значит вопрос в том, была
ли честной монета, выпавшая 22 раза орлом из 26 бросков, — это
[критерий Макнемара](https://ru.wikipedia.org/wiki/Критерий_Макнемара),
посчитанный точно, а не приближением, потому что промоут обычно решается на
нескольких сотнях строк.

Различение, которое он даёт, — ровно то, что нужно перед промоутом: 200
починенных против 3 сломанных это улучшение, 38 против 40 — монета, а правило,
считающее сломанные строки, называет регрессией и то и другое.

Без номеров версий сравнивается то, что в production, с самой новой, — то есть
ровно тот вопрос, который у вас есть перед промоутом.

| Флаг | Что делает |
|---|---|
| `--dataset LABEL` | Прогнать по другому датасету, например по отложенному |
| `-n N` | Остановиться после N строк |
| `--show-changes N` | Напечатать до N строк, где ответы разошлись |
| `--json` | Отдать весь отчёт |
| `--fail-on-regression` | Ненулевой код, если новая ошиблась там, где старая была права |
| `--fail-on-regression significant` | Ненулевой код, только если потери перевешивают приобретения сильнее случайности |
| `--alpha P` | Уровень значимости для режима выше. По умолчанию `0.05` |

```bash
# Курируемый регрессионный набор: терять нельзя ничего.
python manage.py diff reviews.Sentiment --fail-on-regression

# Настоящий датасет перед промоутом: шум пропускаем, настоящую потерю — нет.
python manage.py diff reviews.Sentiment --fail-on-regression significant
```

### Модели, которые обучил не mlango

Сравнению безразлично, откуда взялись две модели: ему нужны два объекта,
умеющих `predict`, и датасет, на котором их прогнать. Значит можно указать на
артефакты, которые у вас уже есть, — без класса `Model` и без перехода на
фреймворк:

```bash
python manage.py diff --dataset reviews.Reviews \
  --left  models/sentiment-v3.joblib \
  --right models/sentiment-v4.joblib
```

Датасет обязателен: сохранённая модель не несёт ни строк, на которых её
оценивать, ни колонки с правильным ответом. Если объявленного датасета ещё нет,
`manage.py inspectdata data/rows.csv` напишет его по файлу.

| Флаг | Что делает |
|---|---|
| `--left URI`, `--right URI` | Две модели. Путь либо `схема:ссылка` |
| `--task` | `classification` (по умолчанию) или `regression` |
| `--target` | Колонка с ответом. По умолчанию — объявленная целевая датасета |
| `--features` | Входные колонки через запятую. По умолчанию — все, кроме целевой и первичного ключа |

Обычный путь читается через joblib с откатом на pickle. Другие схемы приходят из
пакетов, регистрирующихся в группе точек входа `mlango.loaders`:

```toml
[project.entry-points."mlango.loaders"]
mlflow = "my_package.loaders:load_mlflow_model"
```

Функция получает то, что идёт после схемы — `models:/Sentiment/3` для
`mlflow:models:/Sentiment/3`, — и возвращает что угодно с методом `predict`.
Клиенты чужих реестров живут в таких пакетах, а не здесь: фреймворк, который
ставит чужой SDK ради чтения одного файла, — не тот фреймворк, который вам
нужен.

Регрессионные модели сравниваются по расстоянию, а не по равенству — два
вещественных предсказания никогда не равны, — поэтому отчёт даёт среднюю и
максимальную дельту и считает строки, ставшие ближе к истине, против строк,
ставших дальше.

Данные без разметки тоже годятся: тогда отчёт говорит, что изменилось, и не
делает вид, что говорит, что улучшилось.

Этой команды нет в админке, и это намеренно: она загружает две модели и
прогоняет датасет — такое место за командой, которую вы решили запустить, а не за
страницей, открывающейся по клику.

### Слежение за дрейфом

Ушёл ли вход от того, на чём обучалась версия. Читает лог предсказаний, который
выключен, пока вы его не включите, — см. [Мониторинг](monitoring.md).

```bash
python manage.py drift reviews.Sentiment
python manage.py drift reviews.Sentiment --stage production --since 24h
python manage.py drift reviews.Sentiment --against reviews.Incoming
python manage.py drift reviews.Sentiment --since 24h --fail-on significant
```

```
reviews.Sentiment@v4 vs 2841 logged predictions over the last 7d

Column             Kind         PSI     Verdict
-----------------  -----------  ------  -----------
text               text         0.4132  significant
label (predicted)  categorical  0.1801  moderate
```

`--fail-on` завершается ненулевым кодом — именно это делает команду пригодной для
регулярной задачи, а не только для терминала.

### Оценка

```bash
python manage.py evaluate support.AnswerQuality
python manage.py evaluate support.AnswerQuality --show-failures
python manage.py evaluate support.AnswerQuality --min-pass-rate 0.9
```

### Агенты

```bash
python manage.py agent support.Support                          # интерактивно
python manage.py agent support.Support "как мне ...?"            # один запрос
python manage.py agent support.Support "..." --show-steps        # показать вызовы инструментов
python manage.py agent support.Support "..." --session user-42   # с памятью
```

### Что уже произошло

```bash
python manage.py runs list --kind train --status finished -n 20
python manage.py runs show 7c8f1020
python manage.py runs compare 7c8f1020 c089b7e6

python manage.py traces list --agent support.Support
python manage.py traces show a1b2c3d4 -v 2
```

### Разработка

```bash
python manage.py runserver              # 127.0.0.1:8000
python manage.py runserver 8080
python manage.py runserver 0.0.0.0:8080 --reload
python manage.py runserver --no-admin

python manage.py shell                  # IPython, если установлен
python manage.py shell -c "print(Reviews.objects.count())"

python manage.py test                   # pytest, на одноразовом метахранилище
python manage.py test -k splits -x
python manage.py test --coverage
```

`manage.py test` на время прогона переводит метахранилище и хранилище артефактов
во временный каталог, поэтому тест физически не может задеть настоящие данные —
та же идея, что и тестовая база в Django.

`startproject` создаёт готовый каталог `tests/`, так что новый проект зелёный
ещё до первой правки: есть с чего начать и есть что скопировать.

## Общие флаги

Доступны в каждой команде:

| Флаг | Что делает |
|---|---|
| `--settings MODULE` | Использовать другой модуль настроек для этого запуска |
| `-v 0..3` | Тихо, обычно, подробно, очень подробно |
| `--traceback` | Показать полный traceback вместо сообщения |

## Оболочка

`manage.py shell` заранее импортирует все объявленные объекты и несколько
вспомогательных функций:

```python
>>> Reviews.objects.filter(label="positive").count()
1284
>>> Sentiment.versions()
[<ModelVersion reviews.Sentiment@v2 stage=production>, ...]
>>> recent_runs(limit=3)
>>> get_trace("a1b2c3d4").spans
>>> apps.summary()
```

## Свои команды

Положите модуль в `<app>/management/commands/`, и он появится в
`manage.py help` — включая команду, которая **переопределяет встроенную**.
Именно так проект настраивает `train` под себя, не форкая фреймворк.

```python title="reviews/management/commands/import_reviews.py"
from mlango.management import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Импортировать отзывы из хранилища."

    def add_arguments(self, parser):
        parser.add_argument("since", help="Дата в формате ISO, с которой импортировать.")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, **options):
        rows = fetch_since(options["since"])
        if not rows:
            raise CommandError(f"Нечего импортировать с {options['since']}.")

        self.table(
            ["id", "subject"],
            [[r["id"], r["subject"]] for r in rows[:10]],
        )
        if options["dry_run"]:
            self.warn("Пробный запуск: ничего не записано.")
            return
        write(rows)
        self.ok(f"Импортировано отзывов: {len(rows)}.")
```

Что доступно на `self`:

| Метод | Печатает |
|---|---|
| `self.write(msg, level=1)` | Строку, с учётом `-v` |
| `self.ok(msg)` / `self.warn(msg)` | Зелёным / жёлтым |
| `self.stderr(msg)` | В stderr |
| `self.table(headers, rows)` | Выровненную таблицу |
| `self.style.bold(...)` и т. д. | Цвет, отключается при перенаправлении вывода |

Бросайте `CommandError` там, где пользователь должен увидеть сообщение, а не
traceback. Поставьте `requires_apps = False` для команды, которая должна
работать до загрузки приложений, и `requires_settings = False` — для той,
что работает вообще без проекта.

# mlango

**Узнайте, что сломала новая версия модели, — до того как её промоутить.**

*Read this in [English](https://github.com/DrobyshevDev/mlango/blob/master/README.md).*

[![CI](https://github.com/DrobyshevDev/mlango/actions/workflows/ci.yml/badge.svg)](https://github.com/DrobyshevDev/mlango/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mlango)](https://pypi.org/project/mlango/)
[![Python](https://img.shields.io/pypi/pyversions/mlango)](https://pypi.org/project/mlango/)
[![License](https://img.shields.io/pypi/l/mlango)](https://opensource.org/licenses/MIT)

```bash
pip install "mlango[sklearn]"
```

Новая модель точнее на два пункта. Катим?

Агрегированные метрики не скажут, что она заодно сломала сорок строк, которые
раньше работали, — а это обычно ровно те, на которые в прошлом месяце жаловались.
mlango скажет:

```bash
$ python manage.py diff reviews.Sentiment 1 2

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

Точность выросла на три с половиной пункта. Одиннадцать строк, которые раньше
работали, теперь нет — и `broke` единственное место, где это число вообще
появляется. Воспроизвести: [`examples/promotion/`](https://github.com/DrobyshevDev/mlango/tree/master/examples/promotion).

`--fail-on-regression` превращает это в код возврата, который можно поставить
перед промоутом. Та же команда сравнивает **два прогона набора оценок агента**
(у промптов нет номеров версий, поэтому она сравнивает прогоны и говорит, что вы
поменяли), **два файла моделей, которых mlango не обучал**, и **что кандидат
ответил бы на живом трафике**, если запустить его тенью.

Никакой дополнительной инфраструктуры для этого не нужно. Команда читает то, что
фреймворк и так записал.

## Откуда это берётся

ML-проекты превращаются в свалку скриптов: один загружает данные, другой обучает,
где-то лежит ноутбук, выдавший цифру для презентации, и папка `checkpoints/`,
которую уже никто не сопоставит с коммитом. Ничто не знает, что делало остальное,
поэтому сравнить две вещи нечем.

В веб-разработке была ровно такая же проблема, и Django решил её не более удачной
библиотекой, а тем, что стал **фреймворком**: структура проекта, модуль настроек,
декларативные классы, миграции, автоматическая админка и `manage.py`, который всё
связывает.

mlango применяет этот ответ к машинному обучению. Вы объявляете датасеты, модели,
агентов и оценки; фреймворк их запускает, версионирует, записывает — и, раз
записал, может сказать, что изменилось.

```python
# reviews/datasets.py
from mlango.core import fields
from mlango.data import Dataset, JSONLSource

class Reviews(Dataset):
    """Отзывы покупателей о товарах."""

    id = fields.IntegerField()
    text = fields.TextField()
    label = fields.LabelField(["negative", "positive"])

    class Meta:
        source = JSONLSource("data/reviews.jsonl")
        primary_key = "id"
```

```python
# reviews/models.py
from mlango.core import fields
from mlango.training import Model
from reviews.datasets import Reviews

class Sentiment(Model):
    """TF-IDF и логистическая регрессия."""

    max_features = fields.IntegerField(default=20_000, tunable=True)
    C = fields.FloatField(default=1.0, min_value=0.0, tunable=True)

    class Meta:
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
            LogisticRegression(C=self.C),
        )
```

```bash
python manage.py train reviews.Sentiment -p C=2.0
```

Эта одна команда находит ваш класс, открывает отслеживаемый ран, ставит сид всем
генераторам случайных чисел, детерминированно режет данные, вызывает ваш
`build()`, ведёт цикл обучения, пишет метрики, фиксирует git-коммит, сохраняет
артефакт и регистрирует версию модели, готовую к промоуту. Вы написали `build()`
и четыре объявления полей.

---

## Установка

```bash
pip install "mlango[sklearn]"
```

Extras: `sklearn`, `torch`, `anthropic`, `dev`, `docs`, `all`.

## Пять минут с нуля

```bash
mlango startproject myproject
cd myproject
python manage.py migrate
python manage.py train demo.Sentiment
python manage.py runserver

mlango startplugin mlango-lightgbm --kind trainer   # пакет, который поставят другие
```

Откройте <http://127.0.0.1:8000/admin/>. В отличие от пустого каркаса, свежий
проект mlango **уже содержит рабочий пример** — датасет, обученную модель с
настоящими метриками, агента с инструментом и набор оценок. В админке с первого
взгляда есть что смотреть.

Настраивать для этого ничего не нужно: метастор — SQLite, артефакты пишутся в
локальную папку, агенты работают на офлайн-провайдере без API-ключа.

---

## Что вы получаете

### Декларативные классы с `_meta`

Четыре семейства, одна механика. Всё универсальное в фреймворке — админка,
миграции, CLI, схемы API — написано против `_meta`, а не против конкретного
класса. Именно поэтому одна админка отображает все четыре типа объектов.

| Вы объявляете | Вы получаете |
|---|---|
| `Dataset` | Ленивый queryset, валидацию схемы, детерминированные сплиты, версионирование по содержимому |
| `Model` | Гиперпараметры как валидируемые поля, отслеживаемые раны, колбэки, реестр моделей со стадиями |
| `Agent` | Цикл использования инструментов, схемы из аннотаций типов, память, пошаговый трейсинг |
| `Eval` | Пооценочные результаты в метасторе — регрессия становится диффом между двумя ранами |

### QuerySet для данных

Ленивый, композируемый, записываемый рядом с раном, который его использовал:

```python
parts = Reviews.objects.filter(label="positive").shuffle(seed=0).split(train=0.8, val=0.2)

for batch in parts["train"].batch(32):
    ...
```

Лукапы как в Django — `filter(stars__gte=4)`, `exclude(text__icontains="spam")`,
`filter(language__in=["en", "de"])`. Сплиты назначаются по хешу ключа записи,
поэтому **добавление строк никогда не перемещает существующие между train и
test** — именно это свойство делает отложенную выборку по-прежнему честной через
полгода.

### Миграции для схем

```bash
python manage.py makemigrations
python manage.py migrate
```

Изменение полей датасета генерирует настоящий, читаемый файл миграции. Миграции
данных — через `RunPython`, как и ожидается.

### Админка, которую вы не писали

Каждый объявленный объект появляется автоматически, без регистрации.
Регистрируйте только чтобы изменить вид:

```python
@admin.register(Reviews)
class ReviewsAdmin(admin.ObjectAdmin):
    list_display = ("id", "text", "label")
    list_filter = ("label",)
    search_fields = ("text",)
```

Админка показывает предпросмотр данных с фильтрами и поиском, историю ранов с
графиками метрик, сравнение ранов рядом, версии датасетов и моделей с промоутом
в один клик, и пошаговый просмотр каждого вызова агента. Серверный рендеринг,
без сборки и без CDN.

### Агенты как декларации

```python
from mlango.agents import Agent, BufferMemory, tool

@tool
def search_docs(query: str, limit: int = 5) -> list[str]:
    """Поиск по документации продукта.

    Args:
        query: Что искать.
        limit: Максимум результатов.
    """
    return retrieve(query, limit)

class Support(Agent):
    """Отвечает на вопросы о продукте по документации."""

    class Meta:
        model = "claude-opus-5"
        system = "Ты инженер поддержки. Ссылайся на использованные разделы."
        tools = [search_docs]
        memory = BufferMemory(k=20)
```

JSON-схема берётся из аннотаций типов и докстроки, так что инструмент описан
ровно в одном месте. Цикл, повторы, диспетчеризацию инструментов, учёт токенов и
трейсинг берёт на себя фреймворк.

### Развёртывание из той же декларации

```python
# myproject/routes.py
from mlango.serve import path

urlpatterns = [
    path("predict/", Sentiment.as_endpoint(stage="production")),
    path("chat/", Support.as_endpoint()),
]
```

`manage.py runserver` поднимает админку и документированный API вместе;
OpenAPI-схемы выводятся из деклараций, поэтому `/api/docs` описывает входы вашей
модели без единой строки схемы.

---

## Командная строка

```bash
python manage.py check                          # проверить весь проект
python manage.py inspectdata data/reviews.csv    # объявить Dataset по файлу
python manage.py dataset head reviews.Reviews   # заглянуть в данные
python manage.py makemigrations && python manage.py migrate
python manage.py train reviews.Sentiment -p C=2.0 --tag baseline
python manage.py predict reviews.Sentiment "понравилось от начала до конца"
python manage.py explain reviews.Sentiment       # на что опиралась модель
python manage.py drift reviews.Sentiment --since 24h  # сдвинулся ли вход?
python manage.py diff reviews.Sentiment 3 4          # что сломала v4?
python manage.py sweep reviews.Sentiment -p C=0.25,1,4 --promote-best production
python manage.py runs list
python manage.py runs compare 7c8f1020 c089b7e6
python manage.py evaluate reviews.Accuracy --min-pass-rate 0.9
python manage.py agent support.Support           # интерактивная сессия
python manage.py traces show a1b2c3d4            # воспроизвести вызов агента
python manage.py test                            # тесты на временном метасторе
python manage.py shell                           # всё уже импортировано
python manage.py runserver
```

Приложения могут поставлять свои команды в `<app>/management/commands/` — они
появляются в `manage.py help` автоматически, включая переопределение встроенных.

---

## Конфигурация

Один модуль настроек, все значения по умолчанию задокументированы в
`mlango.conf.global_settings`. Бэкенды меняются настройкой, а не переписыванием
кода:

```python
METASTORE = {"URL": "postgresql://user@host/mlango"}   # по умолчанию SQLite
STORAGE = {"BACKEND": "myproject.storage.S3Storage"}
TRAINERS = {"lightgbm": "myproject.trainers.LightGBMTrainer"}
PROVIDERS = {"vllm": "myproject.providers.VLLMProvider"}
SERVE_MIDDLEWARE = ["mlango.serve.middleware.ApiKeyMiddleware", ...]
```

---

## Почему это фреймворк, а не библиотека

Библиотеку вы вызываете. Фреймворк вызывает вас. Эта инверсия — вся суть, и
именно она покупает перечисленные удобства:

- **Структура проекта и настройки** — `manage.py`, `MLANGO_SETTINGS_MODULE`
- **Реестр приложений** — автодискавери `datasets.py`, `models.py`, `agents.py`, `evals.py`, `admin.py`
- **Миграции** — генерируемые, читаемые файлы для объявленных схем
- **Админка из деклараций**
- **Система команд**, которую приложения расширяют и переопределяют
- **Сигналы** — `run_finished`, `epoch_finished`, `tool_called` и другие
- **Подключаемые бэкенды** за настройками

Будь mlango библиотекой, вы бы всё ещё писали цикл обучения, схему трекинга,
админку и CLI. Именно потому, что это фреймворк, их писать не нужно.

---

## Как это устроено

Всё следует из одной мысли: **тело вашего класса компилируется в метаданные, и
каждая универсальная подсистема читает эти метаданные, а не знает про ваш
класс.**

```
          тело вашего класса                 кто это читает
    ┌──────────────────────────┐
    │  class Sentiment(Model): │           ┌──────────────► Страница админки
    │      C = FloatField(…)   │           │
    │                          │           ├──────────────► POST /api/predict/
    │      class Meta:         │  ────►  _meta               и OpenAPI-схема
    │          dataset = …     │        (Options)  │
    │          trainer  = …    │           ├──────────────► Файл миграции
    │                          │           │
    │      def build(self): …  │           ├──────────────► manage.py train
    └──────────────────────────┘           │                manage.py sweep
                                           └──────────────► Eval и реестр версий
```

Ничто справа не импортирует `Model`, `Dataset`, `Agent` или `Eval`. Все читают
`_meta` — поэтому одна админка отображает четыре разных семейства, и поэтому
добавление пятого не потребовало бы её трогать.

### Слои

```
  core            поля · метакласс · Options · реестр · настройки · сигналы
    │             (ничего больше из mlango не импортирует)
    ├── metastore   11 таблиц: раны, метрики, артефакты, версии, трейсы, спаны…
    ├── storage     артефакты за одним узким интерфейсом
    │
    ├── data ─────┐
    ├── training ─┤  четыре семейства. Друг друга не импортируют никогда.
    ├── agents ───┤
    ├── evals ────┘
    │
    └── admin · serve · management     читают всё, но только через _meta
```

Важнее всего среднее правило. Именно из-за того, что семейства не импортируют
друг друга, можно пользоваться агентской половиной без ML-половины, а проект,
объявляющий только датасеты, не загружает ни строчки агентского кода.

### Что на самом деле делает `manage.py train`

```
  train reviews.Sentiment -p C=2.0
        │
        ├─ читает настройки, находит декларации всех приложений
        ├─ разрешает метку в реестре
        ├─ создаёт экземпляр — поля валидируют C=2.0
        ├─ открывает ран: сид, устройство, git-коммит, хост, версия Python
        ├─ режет данные по хешу ключа записи, пишет отпечаток
        ├─ вызывает ваш build(), ведёт цикл, пишет метрики каждую эпоху
        ├─ сохраняет артефакт и регистрирует версию для промоута
        └─ закрывает ран
```

Вы написали `build()` и четыре объявления полей. Всё остальное происходит
независимо от того, вспомнили вы про это или нет, — и это весь аргумент в пользу
фреймворка.

**[Архитектура](https://drobyshevdev.github.io/mlango/ru/architecture/)** —
полная картина: диаграммы последовательностей, схема метастора, каждая точка
расширения с контрактом.
**[Философия](https://drobyshevdev.github.io/mlango/ru/philosophy/)** объясняет
решения, которые за этим стоят.

---

## Документация

Полная документация с учебником, собирающим проект от начала до конца:
**<https://drobyshevdev.github.io/mlango/ru/>**

## Участие

Вклад приветствуется — см. [CONTRIBUTING.md](https://github.com/DrobyshevDev/mlango/blob/master/CONTRIBUTING.md) для настройки
окружения и [CODE_OF_CONDUCT.md](https://github.com/DrobyshevDev/mlango/blob/master/CODE_OF_CONDUCT.md) для правил сообщества.
Переводы документации особенно нужны — по одной странице за раз, см.
[Перевод](https://github.com/DrobyshevDev/mlango/blob/master/docs/translating.md).

## Лицензия

MIT — см. [LICENSE](https://github.com/DrobyshevDev/mlango/blob/master/LICENSE).

mlango не связан с Django Software Foundation и не поддерживается ею. Он с
благодарностью заимствует философию Django.

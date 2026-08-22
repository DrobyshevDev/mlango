# Архитектура

Как устроены части, что что имеет право импортировать и что на самом деле
происходит при запуске команды. Каждая диаграмма здесь — текст в репозитории,
поэтому её видно в диффе.

## Слои

```mermaid
graph TD
    subgraph foundation[" "]
        CORE["<b>core</b><br/><small>поля · метакласс · Options · реестр<br/>настройки · сигналы · исключения</small>"]
    end

    subgraph persistence[" "]
        META["<b>metastore</b><br/><small>11 таблиц · сессии</small>"]
        MIG["<b>migrations</b><br/><small>автодетектор · writer · executor</small>"]
        STORE["<b>storage</b><br/><small>артефакты</small>"]
    end

    subgraph families[" "]
        DATA["<b>data</b><br/><small>Dataset · QuerySet · Sources</small>"]
        TRAIN["<b>training</b><br/><small>Model · тренеры · раны · свипы</small>"]
        AGENTS["<b>agents</b><br/><small>Agent · инструменты · память · провайдеры</small>"]
        EVALS["<b>evals</b><br/><small>Eval · скореры</small>"]
    end

    subgraph surfaces[" "]
        ADMIN["<b>admin</b>"]
        SERVE["<b>serve</b>"]
        CLI["<b>management</b><br/><small>manage.py</small>"]
    end

    CORE --> META
    CORE --> STORE
    META --> MIG
    CORE --> DATA
    CORE --> TRAIN
    CORE --> AGENTS
    CORE --> EVALS
    META -.-> DATA
    META -.-> TRAIN
    META -.-> AGENTS
    META -.-> EVALS
    DATA --> ADMIN
    TRAIN --> ADMIN
    AGENTS --> ADMIN
    EVALS --> ADMIN
    DATA --> SERVE
    TRAIN --> SERVE
    AGENTS --> SERVE
    ADMIN --> CLI
    SERVE --> CLI
```

Правила, которые проверяют и `tests/`, и ревью:

| Слой | Может импортировать | Не должен импортировать |
|---|---|---|
| `core` | стандартную библиотеку | что-либо ещё из mlango |
| `metastore`, `storage` | `core` | четыре семейства |
| `data`, `training`, `agents`, `evals` | `core`, `metastore`, `storage` | **друг друга** |
| `admin`, `serve`, `management` | всё, но через `_meta` | — |

Важнее всего третье. Именно из-за того, что семейства не импортируют друг друга,
можно пользоваться агентской половиной без ML-половины, и поэтому проект,
объявляющий только датасеты, не загружает ни строчки агентского кода.

Помощник, который нужен двум семействам, живёт в `core`. `core/serialization.py`
существует ровно поэтому: `agents` и `evals` оба лезли в приватную функцию
внутри `training`.

## Как декларация становится метаданными

```mermaid
sequenceDiagram
    participant P as Тело вашего класса
    participant M as DeclarativeMeta
    participant O as Options (_meta)
    participant R as Реестр

    P->>M: class Urgency(Model)
    M->>M: собрать Field'ы в порядке объявления
    M->>M: пройти MRO за унаследованными полями и опциями Meta
    M->>O: собрать Options(kind, label, app_label, fields, extras)
    O->>O: сверить ключи Meta с _meta_options
    M->>P: заменить каждое Field на FieldDescriptor
    M->>P: вызвать _prepare() (Dataset получает .objects)
    M->>R: зарегистрировать, если не Meta.abstract
    Note over R: apps.get_model("tickets.Urgency") теперь разрешается
```

Два следствия, о которых стоит знать.

**Опции `Meta` наследуются.** Тела классов в Python сами по себе не наследуются,
поэтому подкласс, написавший свой `class Meta`, молча терял бы всё объявленное
родителем. `Options.inherit_extras()` дозаполняет то, чего подкласс не назвал, —
именно это делает возможным переиспользуемый базовый класс вроде
`TextClassifier`. `abstract` исключён, иначе каждый подкласс абстрактной базы был
бы абстрактным.

**Неизвестный ключ в `Meta` — ошибка.** С перечислением допустимых: молча
проигнорированная опция — это баг, который находят неделями позже.

## Что на самом деле делает `manage.py train`

```mermaid
sequenceDiagram
    autonumber
    participant U as Вы
    participant C as manage.py
    participant R as Реестр
    participant M as Model
    participant Q as QuerySet
    participant T as Тренер
    participant DB as Метастор
    participant S as Хранилище

    U->>C: train tickets.Urgency -p C=2.0
    C->>C: читает настройки, mlango.setup()
    C->>R: находит datasets/models/agents/evals по приложениям
    C->>R: get_model("tickets.Urgency")
    C->>M: Urgency(C=2.0) — поля валидируют значение
    M->>DB: RunContext.start() — сид, устройство, git-коммит, хост
    M->>Q: dataset.objects → сплит по хешу ключа
    M->>DB: пишет _data_fingerprint
    M->>T: fit(train, validation, run, callbacks)
    loop каждая эпоха
        T->>DB: log_metrics()
        T->>C: callbacks.emit(on_epoch_end)
    end
    T->>S: сохраняет обученный артефакт
    M->>DB: регистрирует ModelVersion v1
    M->>DB: ран завершён
```

Всё после пятого шага происходит независимо от того, просили вы об этом или нет.
В этом и состоит инверсия: то, что легко забыть, — это то, чего вы не пишете.

## Метастор

Одиннадцать таблиц, по умолчанию SQLite, та же схема на Postgres.

```mermaid
erDiagram
    RUN ||--o{ METRIC : "записывает"
    RUN ||--o{ ARTIFACT : "пишет"
    RUN ||--o{ MODEL_VERSION : "регистрирует"
    RUN ||--o{ TRACE : "порождает"
    RUN ||--o{ EVAL_RESULT : "оценивает"
    DATASET_VERSION ||--o{ RUN : "был прочитан"
    TRACE ||--o{ SPAN : "по шагам"
    MIGRATION {
        string app
        string name
        datetime applied
    }
    RUN {
        string uuid
        string kind
        string target
        string status
        json params
        json summary
        string git_commit
    }
    MODEL_VERSION {
        int version
        string stage
        string path
    }
    MODEL_VERSION ||--o{ PREDICTION : "ответила на"
    PREDICTION {
        int version
        json inputs
        json output
        datetime created_at
    }
    DATASET_VERSION {
        int version
        string content_hash
        string fingerprint
    }
```

`RUN.kind` — это `train`, `sweep`, `eval` или `agent`. Поэтому свип и его
попытки, или вызов агента и обучение модели, живут в одной истории и одной
админке.

У версии датасета намеренно разделены две идентичности: `fingerprint` — хеш
*декларации*, `content_hash` — хеш *строк*. Изменение схемы и изменение данных —
разные события, и различить их через полгода и есть смысл хранить оба.

## Точки расширения

Каждая — это запись в настройках с путём к классу, поэтому замена любой из них
никогда не означает форк фреймворка.

| Точка | Настройка | Контракт |
|---|---|---|
| Тренер | `TRAINERS` | `fit`, `predict`, `save`, `load` |
| LLM-провайдер | `PROVIDERS` | один метод: `complete()` |
| Хранилище артефактов | `STORAGE["BACKEND"]` | `path`, `open`, `save_bytes`, `read_bytes`, `exists`, `delete`, `size`, `listdir` |
| Middleware сервинга | `SERVE_MIDDLEWARE` | ASGI middleware, снаружи внутрь |
| Колбэки обучения | `DEFAULT_CALLBACKS` | любое подмножество хуков `Callback` |
| Источник данных | `Meta.source` | итерируемое из словарей, по желанию `count()` |
| Команды | `<app>/management/commands/` | `Command(BaseCommand)`, можно переопределить встроенную |

Узкие контракты узки намеренно. У провайдера один метод, потому что цикл агента,
диспетчеризация инструментов, память и трейсинг принадлежат фреймворку: смена
провайдера не должна уметь менять поведение агента.

## Пути запроса

```mermaid
graph LR
    REQ([HTTP-запрос]) --> MW["SERVE_MIDDLEWARE<br/><small>логи · api-ключ · rate limit · guardrails</small>"]
    MW --> R{путь}
    R -->|/admin| ADM["Админка<br/><small>Jinja2, встроенный SVG</small>"]
    R -->|/api/...| EP["Эндпоинт из декларации"]
    R -->|/api/docs| DOC["OpenAPI из _meta"]
    EP --> LOAD["Зарегистрированная версия<br/><small>грузится один раз, кэшируется</small>"]
    ADM --> DB[(Метастор)]
    LOAD --> ST[(Хранилище)]
```

Админка и API — одно ASGI-приложение, поэтому `manage.py runserver` в разработке
это один процесс, а в продакшене тот же объект за gunicorn.

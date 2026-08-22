# Настройки

Один модуль, который берётся из переменной окружения `MLANGO_SETTINGS_MODULE` —
`manage.py` устанавливает её за вас. Все значения по умолчанию лежат в
`mlango.conf.global_settings`; переопределяйте только то, что отличается.

Неизвестная настройка — это ошибка, а не молчаливое «ничего не произошло»:

```python
>>> settings.METASTOR
AttributeError: 'METASTOR' is not a known mlango setting. Settings must be
uppercase and declared in mlango.conf.global_settings.
```

## Ядро

| Настройка | По умолчанию | Зачем |
|---|---|---|
| `BASE_DIR` | текущий каталог | Все относительные пути считаются отсюда |
| `SECRET_KEY` | `""` | Дополнительная энтропия для хеширования. Не храните в git |
| `DEBUG` | `False` | Подробные traceback'и и автоперезагрузка |
| `INSTALLED_APPS` | `[]` | Приложения, чьи объявления загружаются при старте |
| `APP_MODULES` | `("datasets", "models", "agents", "evals", "admin", "signals")` | Модули, которые ищутся в каждом приложении |

## Метахранилище

```python
METASTORE = {
    "URL": "sqlite:///mlango.db",   # или postgresql://user@host/mlango
    "ECHO": False,                  # логировать каждый SQL-запрос
    "POOL_PRE_PING": True,          # только не для SQLite
}
```

Относительный путь к SQLite считается от `BASE_DIR`, поэтому база не «ходит»
за рабочим каталогом оболочки. SQLite работает в режиме WAL, так что админка
может читать, пока цикл обучения пишет.

Переходите на Postgres, когда запуски пишет больше одного воркера.

## Хранилище артефактов

```python
STORAGE = {
    "BACKEND": "mlango.storage.local.LocalStorage",
    "ROOT": "artifacts",
}
```

Сюда попадают чекпоинты, материализованные датасеты и выводы запусков.
Артефакты записываются в метастор относительным именем, поэтому ран, сделанный
на одной машине, разрешается на другой.

```python
STORAGE = {
    "BACKEND": "mlango.storage.s3.S3Storage",
    "ROOT": "s3://my-bucket/mlango",
    # Необязательно — подойдёт любое S3-совместимое: MinIO, R2, B2.
    "ENDPOINT_URL": os.environ.get("S3_ENDPOINT_URL"),
    "REGION": "eu-west-1",
}
```

```bash
pip install "mlango[s3]"
```

Учётные данные — забота boto3, поэтому работают привычные переменные окружения,
роли инстансов и профили, а mlango не держит в руках ни одного секрета, который
может не держать. Для чего-то ещё укажите в `BACKEND` свой подкласс `Storage` —
см. [Обучение на другой машине](serving.md#training-somewhere-else).

## Обучение

| Настройка | По умолчанию | Зачем |
|---|---|---|
| `TRAINERS` | sklearn и torch | `{имя: путь}` — что доступно в `Meta.trainer` |
| `DEVICE` | `"auto"` | `"auto"` выбирает CUDA, если она есть |
| `SEED` | `1337` | Сеет python, numpy и torch в начале каждого запуска |
| `DEFAULT_CALLBACKS` | `[]` | Колбэки, добавляемые к каждому запуску |
| `PREDICTION_LOG` | выключено | Что обслуживаемая модель записывает о запросах |
| `SHADOW` | выключено | Прогонять кандидата рядом с обслуживающей версией |

```python
TRAINERS = {"lightgbm": "myproject.trainers.LightGBMTrainer"}

DEFAULT_CALLBACKS = [
    "mlango.training.callbacks.ProgressBar",
]
```

!!! note
    `DEFAULT_CALLBACKS` только добавляет — запись метрик встроена в сам
    фреймворк, поэтому очистка этого списка никогда не лишит вас истории
    запусков.

```python
PREDICTION_LOG = {
    "ENABLED": True,   # по умолчанию выключено
    "SAMPLE": 0.05,    # оставлять 5% предсказаний
    "MAX_ROWS": 100_000,
}
```

Именно этот лог `manage.py drift` сравнивает с профилем обучающих данных версии.
По умолчанию он выключен, потому что это копия пользовательского ввода в базе —
решение, которое проект принимает сам, а не обнаруживает утром. См.
[Мониторинг](monitoring.md).

```python
SHADOW = {"ENABLED": True, "STAGE": "staging", "SAMPLE": 0.1}
```

Отвечает на каждый запрос дважды — отвечает production, кандидат прогоняется по
тому же входу, — чтобы две версии можно было сравнить на живом трафике. См.
[Теневое развёртывание](serving.md#shadow-deployment).

## Агенты

| Настройка | По умолчанию | Зачем |
|---|---|---|
| `PROVIDERS` | anthropic и echo | `{имя: путь}` — что доступно в `Meta.provider` |
| `DEFAULT_PROVIDER` | `"anthropic"` | Провайдер, если агент не назвал свой |
| `DEFAULT_AGENT_MODEL` | `"claude-opus-5"` | Модель, если агент не назвал свою |
| `DEFAULT_THINKING` | `"adaptive"` | Режим размышлений; `None` не отправляет параметр |
| `AGENT_MAX_STEPS` | `12` | Жёсткий предел цикла вызова инструментов |
| `TRACING` | `True` | Записывать span для каждого вызова модели и инструмента |

Созданный проект стартует на `"echo"`, поэтому работает вообще без ключей.
Переключитесь на `"anthropic"` и задайте `ANTHROPIC_API_KEY`, когда понадобится
настоящая модель.

## Админка

| Настройка | По умолчанию | Зачем |
|---|---|---|
| `ADMIN_ENABLED` | `True` | Подключать админку вообще |
| `ADMIN_URL` | `"/admin"` | Где её смонтировать |
| `ADMIN_SITE_HEADER` | `"mlango administration"` | Текст заголовка |
| `ADMIN_SITE_TITLE` | `"mlango"` | Заголовок вкладки браузера |
| `ADMIN_PAGE_SIZE` | `25` | Строк на страницу по умолчанию |
| `ADMIN_USERNAME` | `"admin"` | Логин для Basic auth |
| `ADMIN_PASSWORD` | `""` | Задайте, чтобы включить Basic auth |

## Обслуживание запросов

| Настройка | По умолчанию | Зачем |
|---|---|---|
| `ROOT_ROUTECONF` | `None` | Модуль с `urlpatterns` |
| `SERVE_MIDDLEWARE` | логирование запросов | Middleware, снаружи внутрь |
| `SERVE_HOST` | `"127.0.0.1"` | Адрес привязки |
| `SERVE_PORT` | `8000` | Порт |
| `SERVE_API_KEYS` | `[]` | Ключи, которые принимает `ApiKeyMiddleware` |
| `SERVE_BLOCKED_TERMS` | `[]` | Что отклоняет `GuardrailMiddleware` |

## Логирование

| Настройка | По умолчанию |
|---|---|
| `LOG_LEVEL` | `"INFO"` |
| `LOG_FORMAT` | `"%(asctime)s %(levelname)-7s %(name)s: %(message)s"` |

## Настройки под окружение

Обычный приём — базовый модуль плюс переопределения:

```python title="myproject/settings/base.py"
INSTALLED_APPS = ["reviews", "support"]
METASTORE = {"URL": "sqlite:///mlango.db"}
```

```python title="myproject/settings/production.py"
import os

from myproject.settings.base import *  # noqa: F403

DEBUG = False
SECRET_KEY = os.environ["MLANGO_SECRET_KEY"]
ADMIN_PASSWORD = os.environ["MLANGO_ADMIN_PASSWORD"]
METASTORE = {"URL": os.environ["DATABASE_URL"]}
STORAGE = {"BACKEND": "mlango.storage.s3.S3Storage", "ROOT": "s3://bucket/mlango"}
DEFAULT_PROVIDER = "anthropic"
SERVE_API_KEYS = os.environ["MLANGO_API_KEYS"].split(",")
```

```bash
python manage.py migrate --settings myproject.settings.production
```

Настройки-словари сливаются со значениями по умолчанию, поэтому, переопределяя
`METASTORE["URL"]`, не нужно заново перечислять `ECHO` и `POOL_PRE_PING`.

## Настройка без модуля

Удобно в ноутбуках и тестах:

```python
from mlango.conf import settings

settings.configure(
    BASE_DIR="/tmp/scratch",
    METASTORE={"URL": "sqlite:///scratch.db"},
    DEFAULT_PROVIDER="echo",
    INSTALLED_APPS=[],
)

import mlango
mlango.setup()
```

Этого достаточно и для команд: `settings.configure()` — полноправный способ
настроиться, поэтому `manage.py`-команды работают и так, без переменной
окружения.

## Проверка конфигурации

```bash
python manage.py check
```

Сообщает, какой модуль настроек выбран, какие приложения установлены, что
объявлено, какой URL у метахранилища и созданы ли его таблицы, какие тренеры и
провайдеры импортируются, разрешается ли каждый путь в `DEFAULT_CALLBACKS`,
`SERVE_MIDDLEWARE` и `STORAGE`, есть ли непримененные миграции и защищена ли
админка паролем.

Разрешать эти пути заранее важно: иначе опечатка в `DEFAULT_CALLBACKS` всплыла
бы как одна и та же ошибка импорта, повторённая по разу на каждую попытку
подбора гиперпараметров.

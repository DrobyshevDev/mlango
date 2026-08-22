# Развёртывание

`manage.py runserver` отдаёт админку и документированный inference API вместе —
так же, как `django-admin runserver` отдаёт админку рядом с вашими вьюхами.

## Маршруты

`routes.py` проекта — это его `urls.py`:

```python title="myproject/routes.py"
from mlango.serve import path

from reviews.models import Sentiment
from support.agents import Support

urlpatterns = [
    path("predict/", Sentiment.as_endpoint(stage="production"), name="sentiment"),
    path("chat/", Support.as_endpoint(), name="support"),
    path("chat/stream/", Support.as_stream_endpoint(), name="support-stream"),
]
```

```python title="myproject/settings.py"
ROOT_ROUTECONF = "myproject.routes"
```

Маршруты монтируются под `/api`, поэтому `path("predict/")` отдаёт
`POST /api/predict/`.

Разнести их по приложениям — через `include`:

```python
from mlango.serve import include, path

urlpatterns = [
    *include("reviews.routes"),
    *include("support.routes"),
]
```

## Эндпоинты моделей

```python
Sentiment.as_endpoint()                     # последняя зарегистрированная версия
Sentiment.as_endpoint(version=3)            # закреплённая
Sentiment.as_endpoint(stage="production")   # та, что промоутнута
```

Версия загружается один раз, лениво, при первом запросе — поэтому запуск сервера
не требует обученной модели, а свежий промоут подхватывается перезапуском.

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H 'Content-Type: application/json' \
  -d '{"input": "отличный фильм"}'

curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H 'Content-Type: application/json' \
  -d '{"inputs": ["отличный фильм", "ужасное кино"], "proba": true}'
```

```json
{
  "model": "reviews.Sentiment",
  "version": 2,
  "predictions": ["positive", "negative"],
  "probabilities": [
    {"negative": 0.04, "positive": 0.96},
    {"negative": 0.95, "positive": 0.05}
  ]
}
```

## Эндпоинты агентов

```bash
curl -X POST http://127.0.0.1:8000/api/chat/ \
  -H 'Content-Type: application/json' \
  -d '{"message": "Как перевыпустить API-ключ?", "session_id": "user-42"}'
```

```json
{
  "agent": "support.Support",
  "output": "Перевыпустите его в Настройки → API-ключи…",
  "steps": 2,
  "trace": "a1b2c3d4…",
  "tools_used": ["search_docs"],
  "usage": {"input_tokens": 1840, "output_tokens": 96, "total_tokens": 1936}
}
```

`session_id` — это то, что даёт агенту непрерывность между запросами через его
бэкенд памяти.

Для интерфейса, который показывает прогресс, есть стриминговый вариант на
Server-Sent Events — см. [Агенты](agents.md#_9).

## Документация бесплатно

Формы запроса и ответа — pydantic-модели, поэтому `/api/docs` описывает каждый
эндпоинт без единой строки OpenAPI:

- Swagger UI — `/api/docs`
- ReDoc — `/api/redoc`
- Схема — `/api/openapi.json`

## Здоровье

```bash
curl http://127.0.0.1:8000/api/health
```

```json
{
  "status": "ok",
  "version": "0.3.0",
  "metastore": true,
  "apps": ["reviews", "support"],
  "counts": {"dataset": 2, "model": 1, "agent": 1, "eval": 1}
}
```

Годится как readiness-проба: подтверждает и что приложение поднялось, *и* что
реестр с метастором доступны.

## Middleware

Стек настраивается в настройках, снаружи внутрь:

```python
SERVE_MIDDLEWARE = [
    "mlango.serve.middleware.RequestLogMiddleware",
    "mlango.serve.middleware.RateLimitMiddleware",
    "mlango.serve.middleware.ApiKeyMiddleware",
    "mlango.serve.middleware.GuardrailMiddleware",
]
```

| Middleware | Делает |
|---|---|
| `RequestLogMiddleware` | Логирует метод, путь, статус и время; добавляет `X-Response-Time-Ms` |
| `ApiKeyMiddleware` | Требует `X-API-Key` из `SERVE_API_KEYS` на маршрутах `/api` |
| `RateLimitMiddleware` | Ограничение в фиксированном окне по адресу клиента |
| `GuardrailMiddleware` | Отклоняет тела запросов с `SERVE_BLOCKED_TERMS` |

Своё — обычным middleware Starlette:

```python
from starlette.middleware.base import BaseHTTPMiddleware


class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request.state.tenant = request.headers.get("X-Tenant", "default")
        return await call_next(request)
```

!!! warning "Ограничения в процессе"
    `RateLimitMiddleware` считает на воркер. Он остановит сорвавшийся скрипт, но
    не заменяет шлюз.

## Ошибки

| Брошено | Статус | Тело |
|---|---|---|
| `ValidationError` | 422 | Сообщения по полям |
| `LookupError` | 404 | Сообщение, например «нет зарегистрированной версии» |
| Любая `MlangoError` | 400 | Сообщение |

Поэтому запрос к модели, которая никогда не обучалась, возвращает 404 с точным
объяснением, а не 500.

## Продакшн

`runserver` — для разработки: один процесс, автоперезагрузка, без управления
воркерами.

`startproject` создаёт `asgi.py` — так же, как он есть в любом проекте Django.
Именно на него направляют продакшн-сервер:

```bash
uvicorn myproject.asgi:application --host 0.0.0.0 --port 8000 --workers 4
gunicorn myproject.asgi:application -k uvicorn.workers.UvicornWorker -w 4
```

`application` собирается при импорте, поэтому реестр заполнен и каждая
объявленная модель разрешается до прихода первого запроса, а не во время него.

### В контейнере

`startproject` создаёт также `Dockerfile`, `.dockerignore` и `compose.yaml`.
Ничего выяснять не нужно:

```bash
docker build -t myproject .
docker run -p 8000:8000 -e MLANGO_SECRET_KEY=... myproject
```

```bash
docker compose up --build      # Postgres под метастор, один веб-процесс
```

Образ двухстадийный, работает не от root, а его `HEALTHCHECK` дёргает
`/api/health` — тот сообщает состав реестра и доступность метастора, поэтому
контейнер, который запустился, но не может разрешить свои настройки, помечается
нездоровым, а не начинает принимать трафик.

`.dockerignore` исключает `mlango.db` и `artifacts/`: скопировать SQLite-файл
разработчика в образ — это и есть способ, которым чужие раны попадают в продакшн.

### Конфигурация приходит из окружения

Сгенерированный `settings.py` читает то, что меняет развёртывание, — чтобы
контейнеру не приходилось править файлы:

| Переменная | Что делает |
|---|---|
| `MLANGO_SETTINGS_MODULE` | Какой модуль настроек загружать |
| `MLANGO_SECRET_KEY` | Перекрывает сгенерированный ключ для разработки |
| `MLANGO_DEBUG=0` | Выключает отладку |
| `DATABASE_URL` | Направляет метастор на Postgres |

Прежде чем выходить в публичный доступ:

- `DEBUG = False`
- `SECRET_KEY` из вашего хранилища секретов
- `ADMIN_PASSWORD`, либо админка за провайдером идентичности
- `SERVE_API_KEYS`, либо аутентификация на шлюзе
- `METASTORE` на Postgres, если раны пишет больше одного воркера
- `STORAGE` на общее хранилище, если воркеры должны видеть артефакты друг друга

## Теневое развёртывание { #shadow-deployment }

Датасет говорит, как кандидат справляется со строками, которые вы отобрали. Он
не может сказать, что кандидат ответил бы людям, которые на самом деле
спрашивали, — а до появления разметки других свидетельств нет.

Промоутните кандидата в `staging`, оставьте production на месте и включите тень:

```python title="myproject/settings.py"
PREDICTION_LOG = {"ENABLED": True, "SAMPLE": 1.0}
SHADOW = {
    "ENABLED": True,
    "STAGE": "staging",   # откуда берётся кандидат
    "SAMPLE": 0.1,        # десятой части запросов обычно достаточно
}
```

```bash
python manage.py train reviews.Sentiment -p C=4.0
python manage.py diff reviews.Sentiment          # стоит ли промоутить по своим данным?
# затем в админке или из Python:
#   Sentiment.promote(5, "staging")
```

После этого каждый запрос получает два ответа: production отвечает вызывающему,
кандидат прогоняется по тому же входу, и оба пишутся в лог под одним id запроса.
Через сутки трафика:

```bash
python manage.py diff reviews.Sentiment 4 5 --from-log --since 24h --show-changes 20
```

```
reviews.Sentiment v4 → v5 on 2841 rows of the prediction log

  agreement      96.8%
  changed        91 row(s)
    neg → pos                61
    pos → neg                30

  The data carries no labels, so this says what changed, not what improved.
```

**Вызывающего это не затрагивает.** Отвечает production; вывод тени уходит в лог
и никогда не приближается к ответу. Упавший кандидат попадает в лог как
предупреждение, а запрос всё равно выполняется: фича, придуманная ради
безопасного промоута, не должна уметь устроить простой.

**Разметки здесь нет, и отчёт об этом говорит.** Продакшн-трафик приходит без
разметки — именно поэтому тень и имеет смысл. `fixed` и `broke` требуют колонки
с истиной и отсутствуют; вы получаете согласие и список запросов, на которые две
версии ответили бы по-разному. Читать надо их.

**Стоит это ровно столько, сколько кажется.** Работают обе версии, поэтому
регулятор — `SAMPLE`: на `0.1` эндпоинт делает 1.1× работы и всё равно набирает
сотни пар в сутки. Кандидат, разрешающийся в ту же версию, что и обслуживающая,
пропускается, а не сравнивается сам с собой — именно это случилось бы на
эндпоинте, отдающем `latest`, сразу после промоута в staging.

## Обучение на другой машине { #training-somewhere-else }

Ноутбук — хорошее место, чтобы объявить модель, и плохое, чтобы её обучить. GPU-
машина становится *частью проекта*, а не тем, откуда вы копируете файлы, за счёт
двух настроек:

```python title="myproject/settings.py"
METASTORE = {"URL": os.environ["DATABASE_URL"]}          # общая история
STORAGE = {
    "BACKEND": "mlango.storage.s3.S3Storage",            # общие артефакты
    "ROOT": "s3://my-bucket/mlango",
}
```

```bash
pip install "mlango[s3]"
```

Дальше — тот же процесс, который вы уже знаете, только выполненный не здесь:

```bash
# на машине с GPU
python manage.py train reviews.Sentiment --tag overnight
```

```bash
# обратно на ноутбуке
python manage.py runs list
python manage.py explain reviews.Sentiment
python manage.py predict reviews.Sentiment "понравилось"
python manage.py runserver
```

Копировать не пришлось ничего. Ран, его метрики, параметры и запись о
воспроизводимости — это строки, которые ноутбук может прочитать; артефакт —
объект, который он может забрать; а админка показывает обучение, за которым вы
не следили.

Работает это потому, что артефакты записываются **относительным именем в
хранилище** — `models/reviews/Sentiment/<run>.joblib`, а не
`/home/gpu/artifacts/models/…`. Ран, записавший абсолютный путь в общий
метастор, оставляет строку, которую может разрешить только одна машина, — тонкий
способ сделать общую базу бесполезной. Версии, зарегистрированные до того, как
это стало правдой, по-прежнему несут абсолютный путь и по-прежнему загружаются —
на той машине, которая их записала.

Опция `ENDPOINT_URL` направляет тот же бэкенд на MinIO, Cloudflare R2 или
Backblaze B2, так что «S3» здесь — это протокол, а не поставщик. Учётные данные
— забота boto3: переменные окружения, роли инстансов, профили. Поэтому у mlango
нет собственных настроек с секретами, которые можно было бы утечь.

!!! note "Чем это не является"
    Здесь нет планировщика задач и поддержки кластеров. mlango не запускает
    обучение — его запускаете вы: по ssh, скриптом Slurm, раннером GitHub
    Actions, чем угодно из того, что у вас уже есть. Фреймворк отвечает за то,
    чтобы результат до вас дошёл.

Полный список дефолтов разработки, которые нужно изменить, — в
[SECURITY.md](https://github.com/DrobyshevDev/mlango/blob/master/SECURITY.md).

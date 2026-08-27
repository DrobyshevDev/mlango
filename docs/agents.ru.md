# Агенты

`Agent` объявляет модель, системный промпт, инструменты и память. Цикл
использования инструментов, диспетчеризацию, учёт токенов и трейсинг берёт на
себя фреймворк.

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
        max_steps = 8
        effort = "high"
```

```python
result = Support().run("Как перевыпустить API-ключ?")
result.output
result.tools_used
result.usage.total_tokens
result.trace_uuid
```

```bash
python manage.py agent support.Support                    # интерактивно
python manage.py agent support.Support "как мне..."        # один вопрос
python manage.py agent support.Support "..." --show-steps
```

## Опции `Meta`

| Опция | По умолчанию | Назначение |
|---|---|---|
| `model` | `DEFAULT_AGENT_MODEL` | Идентификатор модели |
| `system` | `""` | Системный промпт |
| `tools` | `[]` | Функции с `@tool` или экземпляры `Tool` |
| `provider` | `DEFAULT_PROVIDER` | Ключ из настройки `PROVIDERS` |
| `memory` | `NullMemory()` | Бэкенд памяти диалога |
| `max_steps` | `AGENT_MAX_STEPS` | Жёсткий предел цикла |
| `max_tokens` | `4096` | Предел вывода на один вызов модели |
| `thinking` | `"adaptive"` | Режим размышления; `None` убирает параметр |
| `effort` | не задан | `low`, `medium`, `high`, `xhigh`, `max` |
| `tracing` | `TRACING` | Записывать спаны для этого агента |

Собрать промпт из полей самого агента — переопределите `get_system()`:

```python
class Support(Agent):
    tone = fields.ChoiceField(["formal", "friendly"], default="friendly")

    def get_system(self) -> str:
        return f"Ты инженер поддержки. Пиши в тоне: {self.tone}."
```

## Инструменты

Декоратор `@tool` читает аннотации типов и докстроку и строит JSON-схему, которая
нужна модели, — поэтому инструмент описан ровно в одном месте.

```python
@tool
def set_status(ticket: str, status: Literal["open", "closed"]) -> str:
    """Изменить статус тикета.

    Args:
        ticket: Идентификатор тикета, должен начинаться на T-.
        status: Новый статус.
    """
    if not ticket.startswith("T-"):
        raise ToolError("Идентификаторы тикетов начинаются на T-.")
    return f"{ticket} -> {status}"
```

| Аннотация | Схема |
|---|---|
| `str`, `int`, `float`, `bool` | `string`, `integer`, `number`, `boolean` |
| `Literal["a", "b"]` | `string` с `enum` |
| `list[int]` | `array` из `integer` |
| `dict[str, int]` | `object` |
| `X \| None` | `X`, и не обязательный |
| Подкласс `Enum` | `string` с `enum` его значений |
| Параметр со значением по умолчанию | не обязательный, `default` записан |

Записи `Args:` в стиле Google становятся описаниями свойств, включая перенесённые
на несколько строк.

### Ошибки — для модели, а не для вас

Инструмент, бросивший исключение, не роняет агента. Исключение становится
результатом с ошибкой, который модель может прочитать и исправиться:

```python
raise ToolError("Идентификаторы тикетов начинаются на T-.")   # ваше сообщение
raise ValueError("boom")                                      # станет "ValueError: boom"
```

Это намеренно: необработанное исключение посреди цикла теряет весь трейс, а
модель обычно способна исправиться, если ей сказать, что пошло не так.

### Строгий режим

```python
@tool(strict=True)
def book(destination: str, passengers: int) -> str:
    ...
```

Гарантирует, что аргументы точно соответствуют схеме.

## Память

| Бэкенд | Хранит | Переживает перезапуск |
|---|---|---|
| `NullMemory()` | ничего — каждый ран с нуля (по умолчанию) | — |
| `BufferMemory(k=20)` | последние `k` сообщений, в процессе | нет |
| `WindowMemory(k=20, keep_first=1)` | якорный ход плюс последние `k` | нет |
| `MetastoreMemory(max_turns=20)` | восстанавливает из записанных трейсов | да |

`MetastoreMemory` не хранит ничего дополнительно: раз каждый вызов и так
трассируется, диалог восстанавливается из этих записей. Один источник истины,
поэтому просмотр трейсов в админке и память агента не могут разойтись.

Память ключуется по `session_id`:

```python
agent.run("Меня зовут Ада", session_id="user-42")
agent.run("Как меня зовут?", session_id="user-42")
```

## Провайдеры

| Провайдер | Примечания |
|---|---|
| `anthropic` | Claude. Нужен `pip install "mlango[anthropic]"` и учётные данные |
| `echo` | Детерминированный, офлайн. Без учётных данных и без затрат |

Провайдер `echo` — причина, по которой набор тестов фреймворка бесплатно
работает в CI, и по которой свежий проект работает до того, как у вас появится
API-ключ. Он следует простым правилам — `use <tool> {json}` вызывает инструмент —
чего достаточно, чтобы прогнать полный многошаговый цикл.

Переключение в настройках:

```python
DEFAULT_PROVIDER = "anthropic"
DEFAULT_AGENT_MODEL = "claude-opus-5"
```

!!! note "Параметры сэмплирования"
    Текущие модели Claude отклоняют `temperature`, `top_p` и `top_k`. Провайдер
    отбрасывает их с предупреждением, а не даёт запросу упасть; глубина
    управляется через `effort`.

Добавить провайдера — это один класс:

```python
from mlango.agents.providers import Completion, Provider, ToolCall, Usage


class VLLMProvider(Provider):
    name = "vllm"
    requires = ("openai",)

    def complete(self, *, model, messages, system="", tools=None,
                 max_tokens=4096, thinking=None, effort=None, **kw) -> Completion:
        response = call_your_server(...)
        return Completion(
            text=response.text,
            tool_calls=[ToolCall(id=c.id, name=c.name, arguments=c.args)
                        for c in response.calls],
            stop_reason=Completion.TOOL_USE if response.calls else Completion.END_TURN,
            usage=Usage(input_tokens=response.prompt_tokens,
                        output_tokens=response.completion_tokens),
        )
```

```python
PROVIDERS = {"vllm": "myproject.providers.VLLMProvider"}
```

Провайдер делает ровно одну вещь: превращает запрос в один ответ. Цикл, память и
трейсинг остаются за фреймворком, поэтому смена провайдера никогда не меняет
поведение агента.

## Запись и воспроизведение

Набор тестов с агентом внутри — это набор, который ходит в модель: медленно,
платно за вызов и каждый раз по-разному. Ни одного из этих свойств в тестах
быть не должно.

Кассета записывает, что ответил провайдер, и проигрывает это обратно. Она сама
является `Provider`, поэтому агент, цикл, инструменты и трейсинг остаются
настоящими — нет только модели.

```python
from mlango.agents import RecordingProvider, ReplayProvider
from mlango.agents.providers import get_provider

# Один раз, против живого провайдера. Файл закоммитить.
SupportAgent().run(
    "refund please",
    provider=RecordingProvider(get_provider("anthropic"), "cassettes/refund.json"),
)

# Дальше везде — офлайн и одинаково.
SupportAgent().run("refund please", provider=ReplayProvider("cassettes/refund.json"))
```

`provider=` подменяет объявленного только для этого вызова, поэтому запись не
может протечь в следующий тест, как протекла бы настройка.

Вызов сопоставляется по содержимому, так что прогон, шаги которого пошли в
другом порядке, всё равно воспроизведётся. Незаписанный вызов откатывается к
следующей неиспользованной записи — это сохраняет кассету живой после правки
промпта, не изменившей сути вопроса к модели. Передайте `strict=True`, если
предпочитаете, чтобы вам об этом сказали:

```python
ReplayProvider("cassettes/refund.json", strict=True)
```

Файл сохраняется после каждого вызова, а не в конце, поэтому прогон, упавший на
третьем шаге, всё равно записал первые два.

`stream()` воспроизводится из той же кассеты, что и `run()`, и не требует
ничего дополнительно: у них один цикл и один вызов провайдера.

### Почему не кассеты glia

[glia](https://github.com/DrobyshevDev/glia) записывает ту же идею, и делить с
ней файл было бы опрятно. Но в её формате нет места для `Completion.raw_content`
— хода ассистента ровно в том виде, в каком провайдер хочет получить его
обратно, — а потеря этого поля даёт воспроизведения, которые проходят против
кассеты и падают против настоящего API. К тому же glia целиком асинхронна, а
этот слой — нет. Поэтому формат у mlango свой.


## Стриминг

`run()` возвращается только когда цикл закончен. Многошаговый агент может думать
минуту, а пустой экран в течение минуты читается как «сломалось» — поэтому
`stream()` отдаёт события по мере их появления:

```python
from mlango.agents import Finished, TextChunk, ToolCalled

for event in Support().stream("Как перевыпустить API-ключ?"):
    if isinstance(event, TextChunk):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCalled):
        print(f"\n[вызываю {event.name}]")
    elif isinstance(event, Finished):
        print(f"\n[{event.usage['total_tokens']} токенов, трейс {event.trace[:8]}]")
```

| Событие | Когда |
|---|---|
| `Started` | Один раз, перед первым вызовом модели |
| `Thinking` | Перед каждым вызовом модели, чтобы UI показывал прогресс во время ожидания |
| `TextChunk` | Текст ассистента |
| `ToolCalled` | Модель попросила инструмент, он вот-вот запустится |
| `ToolFinished` | Инструмент вернул результат или упал |
| `StepFinished` | Один проход цикла завершён, с учётом токенов |
| `Finished` | Последнее. Несёт тот же `AgentRun`, что возвращает `run()` |
| `Failed` | Цикл бросил исключение; оно последует |

`stream()` и `run()` используют **один** цикл, поэтому они не могут разойтись в
том, что сделал агент. У каждого события есть `.kind` (стабильное snake_case-имя)
и `.describe()`, возвращающий JSON-безопасный объект.

### По HTTP

```python
urlpatterns = [
    path("chat/", Support.as_endpoint()),
    path("chat/stream/", Support.as_stream_endpoint()),
]
```

Стриминговый эндпоинт говорит на `text/event-stream`, который браузеры понимают
нативно:

```javascript
const response = await fetch("/api/chat/stream/", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({message: "Как перевыпустить API-ключ?"}),
});

const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
while (true) {
  const {value, done} = await reader.read();
  if (done) break;
  // строки "event: <kind>" и "data: <json>"
}
```

## Трейсинг

Каждый вызов модели и каждый вызов инструмента становится упорядоченным спаном
одного трейса, поэтому на вопрос «почему он это сказал?» можно ответить потом, из
админки.

```bash
python manage.py traces list
python manage.py traces list --agent support.Support
python manage.py traces show a1b2c3d4 -v 2
```

```python
from mlango.agents import get_trace, recent_traces

trace = get_trace("a1b2c3d4")
[(s.kind, s.name, s.duration_s) for s in trace.spans]
```

Трейсинг работает по принципу best effort: сбой метастора ухудшает наблюдаемость,
но никогда не ломает агента. Выключить для агента — `Meta.tracing = False`,
глобально — `TRACING = False`.

## Цикл

На каждом шаге, пока модель не перестанет просить инструменты или пока не
исчерпан `max_steps`:

1. Вызвать провайдера с сообщениями, системным промптом и схемами инструментов
2. Записать спан `llm` с расходом токенов
3. При `refusal` остановиться и сообщить об этом
4. При `pause_turn` переотправить без изменений, чтобы серверный инструмент
   продолжил
5. Выполнить каждый запрошенный инструмент, каждый в своём спане `tool`
6. Дописать **все** результаты инструментов одним пользовательским сообщением
7. Повторить

Возврат результатов одним сообщением важен: разбиение учит модель перестать
батчить свои вызовы.

## Сигналы

```python
from mlango.core.signals import agent_finished, agent_started, agent_step, tool_called


@receiver(tool_called)
def audit(sender, agent, tool, arguments, **kwargs):
    log.info("%s вызвал %s с %s", sender._meta.label, tool.name, arguments)
```

## Развёртывание

```python
from mlango.serve import path

urlpatterns = [
    path("chat/", Support.as_endpoint()),
]
```

```bash
curl -X POST http://127.0.0.1:8000/api/chat/ \
  -H 'Content-Type: application/json' \
  -d '{"message": "Как перевыпустить API-ключ?", "session_id": "user-42"}'
```

## Безопасность

Инструмент выполняется с правами вашего процесса. Тот, что запускает shell, пишет
файлы или вызывает внутренний API, даёт модели такую же дотяжку. Валидируйте
входы и ставьте всё разрушительное за явное подтверждение на вашей стороне, а не
рассчитывайте на осторожность модели.

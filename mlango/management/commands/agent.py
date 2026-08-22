"""``manage.py agent`` — run a declared agent from the terminal."""

from __future__ import annotations

import sys
from typing import Any

from mlango.core.typing import AgentClass
from mlango.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Send a message to a declared agent, or start an interactive session."

    def add_arguments(self, parser) -> None:
        parser.add_argument("agent", help="Agent label, e.g. support.SupportAgent.")
        parser.add_argument("message", nargs="*", help="The message. Omit for interactive mode.")
        parser.add_argument("--session", default="", help="Session id, for memory continuity.")
        parser.add_argument("--max-steps", type=int, help="Override the step limit.")
        parser.add_argument(
            "--show-steps", action="store_true", help="Print each tool call as it happens."
        )

        # An agent's declaration is its behaviour, so its history is a list of
        # declarations rather than of artifacts.
        parser.add_argument(
            "--versions",
            action="store_true",
            help="List the recorded versions of this agent instead of running it.",
        )
        parser.add_argument(
            "--promote",
            type=int,
            metavar="N",
            help="Move version N to a stage. Use with --stage; default production.",
        )
        parser.add_argument(
            "--stage",
            default="production",
            help="Stage for --promote: none, staging, production, archived.",
        )

    def handle(self, **options: Any) -> None:
        from mlango.core.registry import apps

        agent_class = apps.get_agent(options["agent"])

        if options["promote"] is not None:
            self._promote(agent_class, options)
            return
        if options["versions"]:
            self._versions(agent_class)
            return

        agent = agent_class()

        if options["show_steps"]:
            self._wire_step_output(agent_class)

        message = " ".join(options["message"]).strip()
        if message:
            self._once(agent, message, options)
            return
        self._interactive(agent, options)

    # -- versions ------------------------------------------------------------

    def _versions(self, agent_class: Any) -> None:
        """What the declaration has looked like, and which one is live."""
        versions = agent_class.register_version() and agent_class.versions()
        if not versions:
            self.write(self.style.dim(f"{agent_class._meta.label} has no recorded versions yet."))
            return

        current = agent_class.current_version()
        rows = []
        for version in versions:
            # The marker answers the question people actually have when they
            # open this: which of these is the code in front of me.
            live = "←" if current is not None and current.version == version.version else ""
            rows.append(
                [
                    f"v{version.version}",
                    version.stage,
                    version.fingerprint[:12],
                    ", ".join(version.tools) or "—",
                    version.created_at.strftime("%Y-%m-%d %H:%M"),
                    live,
                ]
            )
        self.table(["Version", "Stage", "Fingerprint", "Tools", "Recorded", "Current"], rows)

        if current is None:
            self.warn(
                "The declaration has changed since the last recorded version — "
                "what is written down and what would run have parted company."
            )

    def _promote(self, agent_class: Any, options: dict[str, Any]) -> None:
        from mlango.core.exceptions import MlangoError

        try:
            version = agent_class.promote(options["promote"], options["stage"])
        except LookupError as exc:
            raise CommandError(str(exc)) from exc
        except MlangoError as exc:
            raise CommandError(str(exc)) from exc

        self.ok(f"{version.ref} is now at stage {version.stage!r}.")
        self.write(
            self.style.dim(
                "A version pins the configuration, not the code: tools are callables "
                "and come from whatever is imported when the agent runs."
            )
        )

    # -- modes ---------------------------------------------------------------

    def _once(self, agent: Any, message: str, options: dict[str, Any]) -> None:
        result = agent.run(
            message, session_id=options["session"], max_steps=options.get("max_steps")
        )
        self.write(result.output)
        self.write("")
        self.write(
            self.style.dim(
                f"steps {result.steps} · tokens {result.usage.total_tokens} "
                f"· trace {result.trace_uuid[:8]}"
            )
        )
        if result.error:
            self.warn(result.error)

    def _interactive(self, agent: Any, options: dict[str, Any]) -> None:
        label = type(agent)._meta.label
        session = options["session"] or "cli"
        self.write(self.style.bold(f"{label} — interactive"))
        self.write(self.style.dim("Type a message, or 'exit' to leave.\n"))

        while True:
            try:
                message = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.write("")
                return
            if message.lower() in {"exit", "quit", ":q"}:
                return
            if not message:
                continue

            result = agent.run(message, session_id=session, max_steps=options.get("max_steps"))
            self.write(f"{label.rpartition('.')[2]}> {result.output}")
            if result.error:
                self.warn(f"  {result.error}")
            self.write(
                self.style.dim(
                    f"  [{result.steps} step(s), {result.usage.total_tokens} tokens, "
                    f"trace {result.trace_uuid[:8]}]"
                )
            )
            self.write("")

    # -- live step output ----------------------------------------------------

    def _wire_step_output(self, agent_class: AgentClass) -> None:
        from mlango.core.signals import tool_called

        style = self.style

        def on_tool(sender, agent, tool, arguments, **kwargs):
            print(style.dim(f"  → {tool.name}({arguments})"), file=sys.stderr)

        # weak=False keeps the closure alive for the length of the command.
        tool_called.connect(on_tool, sender=agent_class, weak=False)

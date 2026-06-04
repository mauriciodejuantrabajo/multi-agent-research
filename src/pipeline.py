"""
Orquestación del equipo: Planner → Researcher → Critic → Writer.

Es el "supervisor" que coordina a los cuatro agentes y maneja el loop de
realimentación. El flujo:

    1. Planner descompone la pregunta en sub-preguntas.
    2. Researcher investiga cada sub-pregunta (comparten un único SourceRegistry,
       así las citas [N] son globales y coherentes en todo el informe).
    3. Critic revisa los hallazgos. Si los aprueba → paso 4. Si no, devuelve
       sub-preguntas de seguimiento y se repite el paso 2 con ellas (hasta un
       máximo de rondas, para no quedar en loop infinito ni gastar la cuota).
    4. Writer redacta el informe final en Markdown.
    5. Se anexa la bibliografía global de fuentes.

Cada transición emite un Event vía el callback `on_event`, para que el CLI o la
UI muestren el progreso en vivo (qué agente está trabajando y en qué).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .agents import Critic, Finding, Planner, Plan, Researcher, Writer
from .llm import DeepSeekClient
from .tools import SourceRegistry, WebTools


@dataclass
class Event:
    """Un hito del pipeline, para mostrar el progreso en vivo."""
    agent: str          # "planner" | "researcher" | "critic" | "writer"
    message: str        # descripción legible
    detail: str = ""    # texto extra opcional (sub-pregunta, veredicto, …)


@dataclass
class ResearchReport:
    question: str
    plan: Plan
    findings: list[Finding]
    report: str
    sources: str
    rounds: int = 1
    events: list[Event] = field(default_factory=list)


class ResearchPipeline:
    def __init__(
        self,
        llm: DeepSeekClient,
        tools: WebTools | None = None,
        max_rounds: int = 2,
        on_event: Callable[[Event], None] | None = None,
    ) -> None:
        # Un único registro de fuentes compartido → citas [N] globales.
        self.tools = tools or WebTools(registry=SourceRegistry())
        self.llm = llm
        self.max_rounds = max_rounds
        self.on_event = on_event

        self.planner = Planner(llm)
        self.researcher = Researcher(self.tools, llm)
        self.critic = Critic(llm)
        self.writer = Writer(llm)

    def _emit(self, events: list[Event], agent: str, message: str, detail: str = "") -> None:
        ev = Event(agent=agent, message=message, detail=detail)
        events.append(ev)
        if self.on_event is not None:
            self.on_event(ev)

    def run(self, question: str) -> ResearchReport:
        events: list[Event] = []

        # 1 · Planner
        self._emit(events, "planner", "Descomponiendo la pregunta…")
        plan = self.planner.plan(question)
        for sq in plan.subquestions:
            self._emit(events, "planner", "Sub-pregunta", sq)

        # 2-3 · Researcher + Critic, con loop de realimentación
        findings: list[Finding] = []
        pending = list(plan.subquestions)
        rounds = 0
        while pending and rounds < self.max_rounds:
            rounds += 1
            for sq in pending:
                self._emit(events, "researcher", "Investigando", sq)
                findings.append(self.researcher.research(sq))

            self._emit(events, "critic", "Revisando los hallazgos…")
            critique = self.critic.review(question, findings)
            for issue in critique.issues:
                self._emit(events, "critic", "Problema detectado", issue)

            if critique.approved or not critique.followups:
                verdict = "Aprobado" if critique.approved else "Sin más para investigar"
                self._emit(events, "critic", verdict)
                break

            # No aprobado: investigar las sub-preguntas de seguimiento en otra ronda.
            self._emit(
                events, "critic", "Pide otra ronda de investigación",
                f"{len(critique.followups)} sub-pregunta(s) nueva(s)",
            )
            pending = critique.followups
        # else del while: agotamos rondas con pendientes; seguimos con lo que hay.

        # 4 · Writer
        self._emit(events, "writer", "Redactando el informe final…")
        report = self.writer.write(question, findings)

        # 5 · Bibliografía global
        sources = self.tools.registry.bibliography()

        return ResearchReport(
            question=question,
            plan=plan,
            findings=findings,
            report=report,
            sources=sources,
            rounds=rounds,
            events=events,
        )

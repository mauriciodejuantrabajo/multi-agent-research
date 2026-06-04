"""
Tests del orquestador (pipeline).

Se mockea el LLM con un cliente falso que responde distinto según el prompt de
sistema de cada agente, y las web tools no tocan la red. Así verificamos el flujo
completo Planner→Researcher→Critic→Writer y el loop de realimentación del Critic,
sin llamadas reales.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline import Event, ResearchPipeline
from src.tools import SourceRegistry, WebTools


class RoleAwareLLM:
    """LLM falso que decide su respuesta según el rol (system prompt) del agente.

    Permite testear el pipeline entero: cada agente recibe la respuesta acorde a
    su rol, y el Researcher puede pedir tools un número configurable de veces.
    """

    def __init__(self, critic_approves: bool = True):
        self.critic_approves = critic_approves
        self.researcher_calls = 0
        self.critic_calls = 0

    def chat(self, messages, tools=None):
        system = messages[0]["content"]
        if system.startswith("Sos el PLANIFICADOR"):
            return {"role": "assistant", "content": '{"topic": "T", "subquestions": ["sq1", "sq2"]}'}

        if system.startswith("Sos el INVESTIGADOR"):
            # Primera vez: pide una búsqueda. Segunda: concluye.
            self.researcher_calls += 1
            if self.researcher_calls % 2 == 1:
                return {
                    "role": "assistant", "content": "",
                    "tool_calls": [{
                        "id": "c", "function": {"name": "web_search", "arguments": {"query": "x"}},
                    }],
                }
            return {"role": "assistant", "content": "- hallazgo concreto [0]."}

        if system.startswith("Sos el CRÍTICO"):
            self.critic_calls += 1
            if self.critic_approves or self.critic_calls > 1:
                return {"role": "assistant", "content": '{"approved": true, "issues": [], "followups": []}'}
            return {
                "role": "assistant",
                "content": '{"approved": false, "issues": ["falta algo"], "followups": ["sq3"]}',
            }

        if system.startswith("Sos el REDACTOR"):
            return {"role": "assistant", "content": "# Informe\n\nResultado final [1]."}

        return {"role": "assistant", "content": ""}


def _tools_no_network():
    tools = WebTools(registry=SourceRegistry())
    return tools


def test_pipeline_happy_path_emits_all_agents():
    llm = RoleAwareLLM(critic_approves=True)
    seen: list[Event] = []
    with patch("src.tools.DDGS") as ddgs_cls:
        ddgs_cls.return_value.__enter__.return_value.text.return_value = []
        pipeline = ResearchPipeline(
            llm, tools=_tools_no_network(), max_rounds=2, on_event=seen.append
        )
        result = pipeline.run("¿pregunta?")

    agents_involved = {e.agent for e in seen}
    assert agents_involved == {"planner", "researcher", "critic", "writer"}
    assert result.report.startswith("# Informe")
    assert result.rounds == 1
    assert len(result.findings) == 2  # sq1 y sq2


def test_pipeline_runs_second_round_when_critic_rejects():
    # El Critic rechaza la 1ra ronda y pide sq3 → debe haber 2 rondas.
    llm = RoleAwareLLM(critic_approves=False)
    with patch("src.tools.DDGS") as ddgs_cls:
        ddgs_cls.return_value.__enter__.return_value.text.return_value = []
        pipeline = ResearchPipeline(llm, tools=_tools_no_network(), max_rounds=2)
        result = pipeline.run("¿pregunta?")

    assert result.rounds == 2
    assert llm.critic_calls == 2
    # sq1, sq2 (ronda 1) + sq3 (ronda 2) = 3 hallazgos
    assert len(result.findings) == 3


def test_pipeline_stops_at_max_rounds():
    # Critic siempre rechaza, pero max_rounds=1 corta tras la primera ronda.
    llm = RoleAwareLLM(critic_approves=False)
    with patch("src.tools.DDGS") as ddgs_cls:
        ddgs_cls.return_value.__enter__.return_value.text.return_value = []
        pipeline = ResearchPipeline(llm, tools=_tools_no_network(), max_rounds=1)
        result = pipeline.run("¿pregunta?")

    assert result.rounds == 1
    assert len(result.findings) == 2  # solo la primera ronda


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

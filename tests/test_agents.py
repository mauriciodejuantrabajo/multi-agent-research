"""
Tests de los agentes y del parser de JSON.

No se hace ninguna llamada de red ni a la API: el LLM se reemplaza por un cliente
falso (FakeLLM) que devuelve mensajes predefinidos, y las web tools se mockean.
Así el CI es reproducible y no consume la cuota de DeepSeek.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents import (
    Critic,
    Finding,
    Planner,
    Researcher,
    Writer,
    _parse_json_block,
)
from src.tools import SourceRegistry, WebTools


class FakeLLM:
    """LLM falso: devuelve, en orden, los mensajes que se le configuran."""

    def __init__(self, scripted_messages):
        self._messages = list(scripted_messages)
        self.calls = 0

    def chat(self, messages, tools=None):
        msg = self._messages[self.calls]
        self.calls += 1
        return msg


def _msg(content="", tool_calls=None):
    m = {"role": "assistant", "content": content}
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    return m


# ---------- _parse_json_block ----------

def test_parse_json_plain_object():
    assert _parse_json_block('{"a": 1}') == {"a": 1}


def test_parse_json_with_code_fence_and_text():
    text = 'Claro, acá tenés:\n```json\n{"topic": "X", "subquestions": ["a"]}\n```\nlisto'
    out = _parse_json_block(text)
    assert out["topic"] == "X"
    assert out["subquestions"] == ["a"]


def test_parse_json_returns_none_on_garbage():
    assert _parse_json_block("no hay json acá") is None
    assert _parse_json_block("") is None


# ---------- Planner ----------

def test_planner_extracts_subquestions():
    llm = FakeLLM([_msg('{"topic": "MCP", "subquestions": ["¿qué es?", "¿por qué importa?"]}')])
    plan = Planner(llm).plan("¿Qué es el MCP?")
    assert plan.topic == "MCP"
    assert plan.subquestions == ["¿qué es?", "¿por qué importa?"]


def test_planner_falls_back_to_question_when_no_subquestions():
    # El modelo devuelve algo inservible → se investiga la pregunta tal cual.
    llm = FakeLLM([_msg("perdón, no entendí")])
    plan = Planner(llm).plan("Pregunta original")
    assert plan.subquestions == ["Pregunta original"]


# ---------- Researcher ----------

def test_researcher_uses_tools_then_concludes():
    # Turno 1: pide leer una URL. Turno 2: concluye con texto + cita.
    scripted = [
        _msg(tool_calls=[{
            "id": "c1",
            "function": {"name": "read_url", "arguments": {"url": "https://x.com"}},
        }]),
        _msg("- El dato clave es 42 [1]."),
    ]
    llm = FakeLLM(scripted)

    tools = WebTools(registry=SourceRegistry())
    fake_resp = MagicMock(status_code=200, text="<title>X</title><p>dato</p>")
    fake_resp.headers = {"Content-Type": "text/html"}
    fake_resp.raise_for_status = MagicMock()

    with patch.object(tools.session, "get", return_value=fake_resp):
        finding = Researcher(tools, llm, max_iters=5).research("¿cuál es el dato?")

    assert "42" in finding.notes
    assert finding.subquestion == "¿cuál es el dato?"
    assert len(tools.registry.sources) == 1  # la URL leída quedó registrada


def test_researcher_respects_max_iters():
    loop = _msg(tool_calls=[{
        "id": "c", "function": {"name": "web_search", "arguments": {"query": "x"}},
    }])
    llm = FakeLLM([loop] * 10)
    tools = WebTools(registry=SourceRegistry())
    with patch("src.tools.DDGS") as ddgs_cls:
        ddgs_cls.return_value.__enter__.return_value.text.return_value = []
        finding = Researcher(tools, llm, max_iters=3).research("loop")
    assert llm.calls == 3
    assert "límite de pasos" in finding.notes


# ---------- Critic ----------

def test_critic_approves():
    llm = FakeLLM([_msg('{"approved": true, "issues": [], "followups": []}')])
    crit = Critic(llm).review("pregunta", [Finding("sq", "notas [1]")])
    assert crit.approved is True
    assert crit.issues == []


def test_critic_requests_followups():
    llm = FakeLLM([_msg(
        '{"approved": false, "issues": ["falta una fuente"], "followups": ["¿y los costos?"]}'
    )])
    crit = Critic(llm).review("pregunta", [Finding("sq", "notas")])
    assert crit.approved is False
    assert crit.followups == ["¿y los costos?"]


def test_critic_fails_open_on_bad_json():
    # Si el JSON no se puede parsear, no trabamos el pipeline: aprobamos.
    llm = FakeLLM([_msg("no es json")])
    crit = Critic(llm).review("pregunta", [Finding("sq", "notas")])
    assert crit.approved is True


# ---------- Writer ----------

def test_writer_returns_markdown():
    llm = FakeLLM([_msg("# Informe\n\nContenido con cita [1].")])
    out = Writer(llm).write("pregunta", [Finding("sq", "notas [1]")])
    assert out.startswith("# Informe")
    assert "[1]" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

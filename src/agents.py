"""
Los cuatro agentes del sistema multi-agente.

Cada agente es un rol con una única responsabilidad y su propio prompt de
sistema. Comparten el mismo cliente LLM, pero se les habla por separado: así
cada uno "piensa" solo en su tarea, sin arrastrar el contexto de los demás.

    Planner     descompone la pregunta en sub-preguntas concretas a investigar.
    Researcher  investiga cada sub-pregunta en la web (web_search + read_url) y
                produce hallazgos con citas [N].
    Critic      revisa los hallazgos: detecta huecos, afirmaciones sin fuente y
                contradicciones; decide si hace falta otra ronda de investigación.
    Writer      redacta el informe final en Markdown a partir de los hallazgos
                aprobados, conservando las citas [N].

El patrón es deliberadamente lineal (Planner → Researcher → Critic → Writer) con
un loop de realimentación Critic→Researcher: es el flujo que se pide en muchas
ofertas (Rozeta) y el más fácil de explicar en una entrevista.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import DeepSeekClient
from .tools import TOOL_SCHEMAS, WebTools


# --------------------------------------------------------------------------- #
# Estructuras de datos que viajan entre agentes
# --------------------------------------------------------------------------- #

@dataclass
class Plan:
    """Lo que produce el Planner: el tema y las sub-preguntas a investigar."""
    topic: str
    subquestions: list[str] = field(default_factory=list)


@dataclass
class Finding:
    """Hallazgo del Researcher para una sub-pregunta."""
    subquestion: str
    notes: str  # texto con afirmaciones y sus citas [N]


@dataclass
class Critique:
    """Veredicto del Critic sobre los hallazgos."""
    approved: bool
    issues: list[str] = field(default_factory=list)
    # Sub-preguntas extra que el Critic pide investigar si no aprueba.
    followups: list[str] = field(default_factory=list)


def _parse_json_block(text: str) -> dict | list | None:
    """Extrae el primer objeto/array JSON de la respuesta del modelo.

    Los modelos a veces envuelven el JSON en ```json ... ``` o lo rodean de
    texto. Esto recorta hasta el primer { o [ y desde el último } o ] para
    tolerar ese ruido sin fallar.
    """
    if not text:
        return None
    start = min(
        (i for i in (text.find("{"), text.find("[")) if i != -1),
        default=-1,
    )
    end = max(text.rfind("}"), text.rfind("]"))
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Agente 1 · Planner
# --------------------------------------------------------------------------- #

PLANNER_PROMPT = """\
Eres el PLANIFICADOR de un equipo de investigación. Tu trabajo es descomponer la
pregunta del usuario en una lista de sub-preguntas concretas e investigables.

Reglas:
- Entre 3 y 5 sub-preguntas, cada una específica y autosuficiente.
- Cubre los distintos ángulos del tema (qué, cómo, por qué, ejemplos, críticas).
- No respondas la pregunta: solo descompónla.
- Escribe las sub-preguntas en el MISMO idioma que la pregunta del usuario.

Responde SOLO con un objeto JSON, sin texto alrededor, con esta forma exacta:
{"topic": "<reformulación breve del tema>", "subquestions": ["...", "..."]}
"""


class Planner:
    def __init__(self, llm: DeepSeekClient) -> None:
        self.llm = llm

    def plan(self, question: str) -> Plan:
        messages = [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": question},
        ]
        msg = self.llm.chat(messages)
        data = _parse_json_block(msg.get("content", "")) or {}
        subs = [str(s).strip() for s in data.get("subquestions", []) if str(s).strip()]
        # Fallback defensivo: si el modelo no dio sub-preguntas, investigamos el tema tal cual.
        if not subs:
            subs = [question.strip()]
        return Plan(topic=str(data.get("topic", question)).strip() or question, subquestions=subs)


# --------------------------------------------------------------------------- #
# Agente 2 · Researcher (el único con herramientas)
# --------------------------------------------------------------------------- #

RESEARCHER_PROMPT = """\
Eres el INVESTIGADOR del equipo. Te dan UNA sub-pregunta y tienes que responderla
investigando en la web.

Herramientas:
- web_search(query): busca y devuelve resultados (título, URL, snippet).
- read_url(url): descarga una página y devuelve su texto.

Cómo trabajar:
- Busca con web_search y lee con read_url al menos 1-2 páginas relevantes antes
  de concluir.
- No inventes datos: afirma solo lo que viste en una página que leíste.
- Cuando lees una página, su contenido empieza con "[Fuente N]". Usa ese número
  para citar: pon [N] justo después de cada afirmación tomada de esa fuente.
- Al terminar, escribe tus hallazgos para esta sub-pregunta en el mismo idioma
  que la pregunta: viñetas con datos concretos, cada uno con su cita [N]. Sé
  breve y factual.
- No agregues la lista de fuentes: el sistema la anexa al final del informe.
"""


class Researcher:
    """Investiga una sub-pregunta con un loop ReAct sobre las web tools."""

    def __init__(self, tools: WebTools, llm: DeepSeekClient, max_iters: int = 6) -> None:
        self.tools = tools
        self.llm = llm
        self.max_iters = max_iters
        self._dispatch = {
            "web_search": self.tools.web_search,
            "read_url": self.tools.read_url,
        }

    def _run_tool(self, name: str, args: dict) -> str:
        fn = self._dispatch.get(name)
        if fn is None:
            return f"Error: herramienta desconocida '{name}'."
        try:
            return fn(**args)
        except TypeError as exc:
            return f"Error: argumentos inválidos para '{name}': {exc}"
        except Exception as exc:  # noqa: BLE001 — el resultado vuelve al modelo
            return f"Error al ejecutar '{name}': {exc}"

    def research(self, subquestion: str) -> Finding:
        messages = [
            {"role": "system", "content": RESEARCHER_PROMPT},
            {"role": "user", "content": f"Sub-pregunta a investigar:\n{subquestion}"},
        ]
        for _ in range(self.max_iters):
            msg = self.llm.chat(messages, tools=TOOL_SCHEMAS)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return Finding(subquestion=subquestion, notes=msg.get("content", "").strip())

            messages.append(msg)
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {}) or {}
                if isinstance(args, str):  # DeepSeek/OpenAI mandan los args como JSON string
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                result = self._run_tool(name, args)
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": call.get("id", ""),
                    "name": name,
                })

        return Finding(
            subquestion=subquestion,
            notes="(No llegué a concluir esta sub-pregunta dentro del límite de pasos.)",
        )


# --------------------------------------------------------------------------- #
# Agente 3 · Critic
# --------------------------------------------------------------------------- #

CRITIC_PROMPT = """\
Eres el CRÍTICO del equipo. Te dan la pregunta original y los hallazgos del
investigador. Tu trabajo es controlar la CALIDAD antes de escribir el informe.

Revisa:
- ¿Hay afirmaciones importantes SIN cita [N]?
- ¿Quedan ángulos de la pregunta sin cubrir (huecos)?
- ¿Hay contradicciones entre hallazgos?

Sé exigente pero razonable: si los hallazgos alcanzan para un buen informe,
APRUEBA. Solo pide más investigación si falta algo realmente importante.

Responde SOLO con un objeto JSON, sin texto alrededor:
{"approved": true|false, "issues": ["..."], "followups": ["nueva sub-pregunta", "..."]}
- "issues": problemas detectados (vacío si está todo bien).
- "followups": sub-preguntas extra a investigar (solo si approved=false).
"""


class Critic:
    def __init__(self, llm: DeepSeekClient) -> None:
        self.llm = llm

    def review(self, question: str, findings: list[Finding]) -> Critique:
        findings_text = "\n\n".join(
            f"### {f.subquestion}\n{f.notes}" for f in findings
        )
        user = (
            f"Pregunta original:\n{question}\n\n"
            f"Hallazgos del investigador:\n{findings_text}"
        )
        messages = [
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": user},
        ]
        msg = self.llm.chat(messages)
        data = _parse_json_block(msg.get("content", "")) or {}
        # Si no se pudo parsear, aprobamos para no trabar el pipeline (fail-open).
        approved = bool(data.get("approved", True))
        issues = [str(x).strip() for x in data.get("issues", []) if str(x).strip()]
        followups = [str(x).strip() for x in data.get("followups", []) if str(x).strip()]
        return Critique(approved=approved, issues=issues, followups=followups)


# --------------------------------------------------------------------------- #
# Agente 4 · Writer
# --------------------------------------------------------------------------- #

WRITER_PROMPT = """\
Eres el REDACTOR del equipo. Te dan la pregunta original y los hallazgos ya
validados por el crítico. Escribe un INFORME final en Markdown, en el mismo
idioma que la pregunta original.

Estructura:
- Un título con #.
- Una introducción breve (2-3 líneas).
- Secciones con ## que organicen el contenido de forma lógica.
- Una conclusión.

Reglas:
- Usa SOLO la información de los hallazgos. No agregues datos nuevos.
- Conserva las citas [N] exactamente donde corresponden a cada afirmación.
- No inventes ni renumeres las citas.
- No agregues la lista de fuentes al final: el sistema la anexa automáticamente.
"""


class Writer:
    def __init__(self, llm: DeepSeekClient) -> None:
        self.llm = llm

    def write(self, question: str, findings: list[Finding]) -> str:
        findings_text = "\n\n".join(
            f"### {f.subquestion}\n{f.notes}" for f in findings
        )
        user = (
            f"Pregunta original:\n{question}\n\n"
            f"Hallazgos validados:\n{findings_text}"
        )
        messages = [
            {"role": "system", "content": WRITER_PROMPT},
            {"role": "user", "content": user},
        ]
        msg = self.llm.chat(messages)
        return msg.get("content", "").strip()

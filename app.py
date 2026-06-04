"""
Interfaz web (Streamlit) del sistema multi-agente de investigación.

Escribes una pregunta y ves, en vivo, cómo trabaja el equipo de agentes
(Planner → Researcher → Critic → Writer). Al final se muestra el informe en
Markdown con su bibliografía y un botón para descargarlo.

    streamlit run app.py
"""

from __future__ import annotations

from dotenv import load_dotenv

import streamlit as st

from src.llm import LLMError, get_client
from src.pipeline import Event, ResearchPipeline
from src.tools import SourceRegistry, WebTools

load_dotenv()

st.set_page_config(page_title="Multi-Agent Research", page_icon="🧠", layout="centered")

# Ícono y color por agente, igual que en el CLI.
AGENT_META = {
    "planner": ("🧭", "Planner"),
    "researcher": ("🔎", "Researcher"),
    "critic": ("🧐", "Critic"),
    "writer": ("✍️", "Writer"),
}

st.title("🧠 Multi-Agent Research")
st.caption(
    "Un equipo de 4 agentes —**Planner → Researcher → Critic → Writer**— investiga "
    "tu pregunta en la web y redacta un informe con citas verificables."
)

with st.sidebar:
    st.header("Cómo funciona")
    st.markdown(
        "1. **🧭 Planner** descompone la pregunta en sub-preguntas.\n"
        "2. **🔎 Researcher** investiga cada una en la web (busca y lee páginas).\n"
        "3. **🧐 Critic** revisa la calidad y pide otra ronda si falta algo.\n"
        "4. **✍️ Writer** redacta el informe final conservando las citas [N]."
    )
    max_rounds = st.slider(
        "Rondas máximas de investigación", min_value=1, max_value=3, value=2,
        help="Cuántas veces el Critic puede pedir que el Researcher profundice.",
    )

question = st.text_input(
    "Tu pregunta de investigación",
    placeholder="Ej: ¿Qué es el Model Context Protocol y por qué importa?",
)
go = st.button("Investigar", type="primary", disabled=not question.strip())

if go and question.strip():
    # Un pipeline (y registro de fuentes) nuevo por consulta.
    tools = WebTools(registry=SourceRegistry())
    llm = get_client()

    progress_box = st.container()
    rendered: list[str] = []

    def on_event(event: Event) -> None:
        icon, name = AGENT_META.get(event.agent, ("•", event.agent))
        line = f"{icon} **{name}**: {event.message}"
        if event.detail:
            line += f" — _{event.detail}_"
        rendered.append(line)
        # Re-render acumulado: se ve el avance paso a paso.
        progress_box.markdown("\n\n".join(rendered))

    pipeline = ResearchPipeline(llm, tools=tools, max_rounds=max_rounds, on_event=on_event)

    with st.status("El equipo está trabajando…", expanded=True) as status:
        try:
            result = pipeline.run(question.strip())
            status.update(label="Investigación completa ✅", state="complete")
        except LLMError as exc:
            status.update(label="Error", state="error")
            st.error(f"Error de LLM: {exc}")
            st.stop()

    st.divider()
    st.subheader("📄 Informe")
    st.markdown(result.report)

    if result.sources:
        with st.expander("📚 Fuentes", expanded=True):
            st.text(result.sources)

    download = result.report + (f"\n\n---\n\n{result.sources}" if result.sources else "")
    st.download_button(
        "⬇️ Descargar informe (.md)",
        data=download,
        file_name="informe.md",
        mime="text/markdown",
    )
    st.caption(f"Rondas de investigación: {result.rounds}")

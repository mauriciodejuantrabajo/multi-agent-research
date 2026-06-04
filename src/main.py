"""
CLI del sistema multi-agente de investigación.

Uso:
    python -m src.main "¿Qué es el Model Context Protocol y por qué importa?"
    python -m src.main                       # modo interactivo
    python -m src.main "tema" -o informe.md  # guarda el informe en un archivo

Muestra en vivo qué agente está trabajando (Planner/Researcher/Critic/Writer) y,
al final, el informe en Markdown con su bibliografía.
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .llm import LLMError, get_client
from .pipeline import Event, ResearchPipeline

# En Windows la consola suele usar cp1252 y rompe con los emojis de la traza.
# Forzamos UTF-8 en la salida para que el CLI funcione en cualquier terminal.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

console = Console()

# Color e ícono por agente, para que la traza se lea de un vistazo.
_AGENT_STYLE = {
    "planner": ("🧭", "magenta"),
    "researcher": ("🔎", "cyan"),
    "critic": ("🧐", "yellow"),
    "writer": ("✍️", "green"),
}


def _show_event(event: Event) -> None:
    icon, color = _AGENT_STYLE.get(event.agent, ("•", "white"))
    name = event.agent.capitalize()
    line = f"[{color}]{icon} {name}[/{color}]: {event.message}"
    if event.detail:
        line += f" [dim]— {event.detail}[/dim]"
    console.print(line)


def run_once(pipeline: ResearchPipeline, question: str, out_path: str | None) -> None:
    try:
        result = pipeline.run(question)
    except LLMError as exc:
        console.print(f"[red]Error de LLM:[/red] {exc}")
        return

    console.print()
    console.print(Panel(Markdown(result.report), title="Informe", border_style="green"))
    if result.sources:
        console.print(Panel(result.sources, title="Fuentes", border_style="cyan"))

    if out_path:
        full = result.report
        if result.sources:
            full += f"\n\n---\n\n{result.sources}"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(full)
        console.print(f"[dim]Informe guardado en {out_path}[/dim]")


def build_pipeline() -> ResearchPipeline:
    llm = get_client()
    return ResearchPipeline(llm, on_event=_show_event)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sistema multi-agente de investigación (Planner→Researcher→Critic→Writer)."
    )
    parser.add_argument(
        "question", nargs="?", default=None,
        help="Pregunta a investigar; si se omite, se entra en modo interactivo.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Ruta de archivo donde guardar el informe en Markdown.",
    )
    args = parser.parse_args()

    load_dotenv()

    if args.question:
        run_once(build_pipeline(), args.question, args.output)
        return

    console.print(Panel.fit(
        "[bold]Multi-Agent Research[/bold]\n"
        "Un equipo de 4 agentes investiga tu pregunta: planifica, investiga en la "
        "web, se autocritica y redacta un informe con citas.\n"
        "[dim]Escribe tu pregunta. 'salir' para terminar.[/dim]",
        border_style="blue",
    ))

    while True:
        try:
            question = console.input("\n[bold cyan]?[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n¡Hasta luego!")
            break
        if not question:
            continue
        if question.lower() in {"salir", "exit", "quit", "q"}:
            console.print("¡Hasta luego!")
            break
        run_once(build_pipeline(), question, None)


if __name__ == "__main__":
    main()

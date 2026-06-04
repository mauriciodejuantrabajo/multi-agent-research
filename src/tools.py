"""
Herramientas del agente de investigación web.

Dos herramientas, igual que un humano investigando:
    - web_search(query): busca en la web y devuelve resultados (título, URL, snippet).
    - read_url(url):      descarga una página y la convierte a texto limpio.

La búsqueda usa DuckDuckGo (sin API key, alineado con la premisa "sin APIs de pago").
Cada herramienta registra las URLs que va viendo en `SourceRegistry`, para que el
agente pueda citar las fuentes con [1], [2], … al final.

Hay límites de tamaño (igual que en file-system-agent) para no inundar el contexto
del modelo con páginas enormes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

try:
    # Paquete actual del wrapper de DuckDuckGo.
    from ddgs import DDGS
except ImportError:  # pragma: no cover - nombre antiguo del mismo paquete
    from duckduckgo_search import DDGS  # type: ignore


# Límites para no inundar el contexto del modelo.
MAX_SEARCH_RESULTS = 6
MAX_PAGE_CHARS = 8_000
REQUEST_TIMEOUT = 15

# User-Agent de navegador: muchos sitios bloquean el default de requests.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Etiquetas cuyo contenido no aporta al texto principal de la página.
_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]


@dataclass
class Source:
    """Una fuente vista durante la investigación, para citarla luego."""
    url: str
    title: str


@dataclass
class SourceRegistry:
    """Lleva el orden de las fuentes leídas para numerarlas como [1], [2], …"""
    sources: list[Source] = field(default_factory=list)

    def add(self, url: str, title: str) -> int:
        """Registra una fuente (sin duplicar) y devuelve su número de cita (1-based)."""
        for i, src in enumerate(self.sources, start=1):
            if src.url == url:
                return i
        self.sources.append(Source(url=url, title=title or url))
        return len(self.sources)

    def bibliography(self) -> str:
        """Lista numerada de fuentes para mostrar al final de la respuesta."""
        if not self.sources:
            return ""
        lines = [f"[{i}] {s.title}\n    {s.url}" for i, s in enumerate(self.sources, start=1)]
        return "Fuentes:\n" + "\n".join(lines)


class WebTools:
    def __init__(self, registry: SourceRegistry | None = None) -> None:
        self.registry = registry or SourceRegistry()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    # ---- herramientas ----
    def web_search(self, query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
        """Busca en la web y devuelve los mejores resultados (título, URL, snippet)."""
        query = (query or "").strip()
        if not query:
            return "Error: la búsqueda está vacía."
        n = max(1, min(int(max_results or MAX_SEARCH_RESULTS), MAX_SEARCH_RESULTS))
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=n))
        except Exception as exc:  # noqa: BLE001 — el resultado vuelve al modelo
            return f"Error al buscar '{query}': {exc}"

        if not results:
            return f"Sin resultados para '{query}'."

        lines = [f"Resultados para '{query}':"]
        for r in results:
            title = r.get("title", "").strip()
            url = r.get("href") or r.get("url") or ""
            snippet = (r.get("body") or "").strip()
            lines.append(f"- {title}\n  URL: {url}\n  {snippet[:200]}")
        return "\n".join(lines)

    def read_url(self, url: str) -> str:
        """Descarga una página web y devuelve su texto principal (limpio y truncado)."""
        url = (url or "").strip()
        if not re.match(r"^https?://", url):
            return f"Error: URL inválida '{url}' (debe empezar con http:// o https://)."
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            return f"Error al leer '{url}': {exc}"

        content_type = resp.headers.get("Content-Type", "")
        if "html" not in content_type and "text" not in content_type:
            return f"Error: '{url}' no es una página de texto ({content_type})."

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        title = (soup.title.string or "").strip() if soup.title else url
        text = soup.get_text(separator="\n")
        # Colapsar líneas en blanco para compactar el texto.
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()

        cite = self.registry.add(url, title)
        truncated = len(text) > MAX_PAGE_CHARS
        body = text[:MAX_PAGE_CHARS]
        suffix = "\n… (truncado)" if truncated else ""
        return (
            f"[Fuente {cite}] {title}\nURL: {url}\n\n{body}{suffix}"
        )


# --- Esquemas de las tools en el formato que espera DeepSeek/OpenAI ---
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Busca en la web y devuelve una lista de resultados (título, URL, "
                "snippet). Usar para descubrir qué páginas leer sobre un tema."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta de búsqueda en lenguaje natural.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Cantidad máxima de resultados (1-6).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": (
                "Descarga una página web y devuelve su texto principal limpio. "
                "Usar para leer en detalle una URL encontrada con web_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL completa de la página a leer (http/https).",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

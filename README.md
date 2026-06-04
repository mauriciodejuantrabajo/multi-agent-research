# 🧠 Multi-Agent Research

Haz una pregunta y un **equipo de cuatro agentes** investiga por ti: uno
**planifica**, otro **investiga en la web**, otro **se autocritica** y un último
**redacta** un informe en Markdown **con citas verificables a las fuentes**.

Es el patrón clásico de sistemas multi-agente —**Planner → Researcher → Critic →
Writer**— con un *loop de realimentación*: si el crítico detecta huecos, manda al
investigador a profundizar antes de escribir. Usa **tool-calling nativo** de la
API de **DeepSeek**.

Funciona con **interfaz web** (Streamlit, progreso en vivo) o por **CLI**.

```
?  ¿Qué es el Model Context Protocol y por qué importa?

  🧭 Planner: Descomponiendo la pregunta…
  🧭 Planner: Sub-pregunta — ¿Qué es el Model Context Protocol?
  🧭 Planner: Sub-pregunta — ¿Qué problema resuelve y por qué importa?
  🔎 Researcher: Investigando — ¿Qué es el Model Context Protocol?
  🔎 Researcher: Investigando — ¿Qué problema resuelve y por qué importa?
  🧐 Critic: Revisando los hallazgos…
  🧐 Critic: Aprobado
  ✍️ Writer: Redactando el informe final…

  # El Model Context Protocol (MCP)
  El MCP es un protocolo abierto que estandariza cómo las aplicaciones proveen
  contexto a los modelos de lenguaje [1]. …

  Fuentes
  [1] Introduction - Model Context Protocol — modelcontextprotocol.io
  [2] Introducing the Model Context Protocol — anthropic.com
```

## El problema

Una investigación seria no es una sola llamada al modelo: hay que **descomponer**
la pregunta, **buscar y leer** varias fuentes, **contrastar**, detectar **huecos**
y recién entonces **redactar**. Pedírselo todo a un único prompt produce
respuestas planas, sin estructura y con riesgo de alucinación.

## La solución

**Dividir el trabajo en roles**, como un equipo humano. Cada agente tiene una
única responsabilidad y su propio prompt, así "piensa" solo en su tarea:

| Agente | Rol | Herramientas |
|--------|-----|--------------|
| 🧭 **Planner** | Descompone la pregunta en 3-5 sub-preguntas investigables | — |
| 🔎 **Researcher** | Investiga cada sub-pregunta (busca y lee páginas) con citas [N] | `web_search`, `read_url` |
| 🧐 **Critic** | Revisa calidad: huecos, afirmaciones sin fuente, contradicciones | — |
| ✍️ **Writer** | Redacta el informe final en Markdown conservando las citas | — |

El **Critic** es la pieza clave: si no aprueba los hallazgos, devuelve
**sub-preguntas de seguimiento** y el Researcher hace otra ronda (hasta un máximo
configurable). Es realimentación, no un pipeline ciego.

Todos los investigadores comparten un único **registro de fuentes**, así las
citas `[N]` son **globales y coherentes** en todo el informe, y la bibliografía
se anexa automáticamente al final (cada afirmación queda **verificable**).

## Características

- 👥 **Cuatro roles especializados** con prompts independientes (separación de responsabilidades).
- 🔁 **Loop de realimentación Critic → Researcher**: profundiza si falta algo, con tope de rondas.
- 📚 **Informe con citas** verificables `[N]` + bibliografía global automática.
- 🖥️ **Interfaz web** (Streamlit): muestra en vivo qué agente trabaja y en qué; descarga el `.md`.
- 💻 **CLI** (rich) con traza en vivo y opción de guardar el informe en un archivo.
- 🧪 **Tests sin red**: el CI no consume la cuota de API.

## Arquitectura

```
app.py             Interfaz web (Streamlit): progreso en vivo + descarga del informe.
src/
├── llm.py         Cliente DeepSeek (API compatible OpenAI): tool-calling.
├── tools.py       web_search + read_url + SourceRegistry (numera y deduplica citas).
├── agents.py      Los 4 agentes: Planner, Researcher, Critic, Writer.
├── pipeline.py    Orquestador: coordina los agentes y el loop de realimentación.
└── main.py        CLI (rich) con traza en vivo y export a Markdown.
tests/             Tests con mocks (sin red real ni llamadas a la API).
```

El **flujo** que coordina `pipeline.py`:

```
pregunta
   │
   ▼
🧭 Planner ──► [sub-preguntas]
   │
   ▼
🔎 Researcher ──► [hallazgos con citas]  ◄──┐
   │                                        │ otra ronda
   ▼                                        │ (sub-preguntas
🧐 Critic ──► ¿aprueba? ── no ──────────────┘  de seguimiento)
   │ sí
   ▼
✍️ Writer ──► informe.md + bibliografía
```

## Requisitos

- **Python 3.10+**
- Una **API key de DeepSeek** → se obtiene en https://platform.deepseek.com

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/mauriciodejuantrabajo/multi-agent-research.git
cd multi-agent-research

# 2. (Opcional) crear un entorno virtual
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar la API key (ver siguiente sección)
```

## Configuración

Copia la plantilla de variables de entorno y completa tu API key:

```bash
cp .env.example .env       # en Windows: copy .env.example .env
```

Edita `.env` y coloca tu key de DeepSeek:

```env
DEEPSEEK_API_KEY=sk-tu-key-real-aca
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

> 🔒 **El archivo `.env` está en `.gitignore` y nunca se sube al repositorio.** Tu
> API key queda solo en tu máquina. El archivo versionado es `.env.example`, que
> solo contiene un placeholder (`sk-...`).

## Uso

### Interfaz web (recomendada)

```bash
streamlit run app.py
```

Se abre en `http://localhost:8501`. Escribe tu pregunta y observa cómo el equipo
trabaja paso a paso; al final obtienes el informe con sus fuentes y un botón para
descargarlo. En el panel lateral puedes ajustar las **rondas máximas** de
investigación.

### CLI

```bash
python -m src.main "¿Qué es el Model Context Protocol y por qué importa?"
python -m src.main                                    # modo interactivo
python -m src.main "tema a investigar" -o informe.md  # guarda el informe
```

Verás la traza en vivo de cada agente y, al final, el informe en Markdown con su
bibliografía. Escribe `salir` para terminar el modo interactivo.

## El modelo

Se usa la API de **DeepSeek** (formato compatible con OpenAI). El modelo es
configurable en `.env` sin tocar código:

```env
DEEPSEEK_MODEL=deepseek-v4-flash
```

Si tu cuenta tiene otro modelo, basta con colocar su identificador exacto en
`DEEPSEEK_MODEL`.

## Tests

```bash
pytest
```

Los tests reemplazan el LLM por uno falso (que responde según el rol de cada
agente) y mockean las web tools: se cubre el flujo completo Planner→Researcher→
Critic→Writer **y** el loop de realimentación del crítico. **No se hace ninguna
llamada de red real**, así el CI es reproducible y no consume la cuota de API.

## Licencia

[MIT](LICENSE) © Mauricio De Juan

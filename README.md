> **Idioma / Language:** **English** · [Español](README.es.md)

# 🧠 Multi-Agent Research

Ask a question and a **team of four agents** researches for you: one **plans**,
one **researches the web**, one **self-critiques** and a last one **writes** a
Markdown report **with verifiable citations to the sources**.

It's the classic multi-agent pattern —**Planner → Researcher → Critic →
Writer**— with a *feedback loop*: if the critic detects gaps, it sends the
researcher back to dig deeper before writing. It uses **native tool-calling**
from the **DeepSeek** API.

It works with a **web interface** (Streamlit, live progress) or via **CLI**.

```
?  What is the Model Context Protocol and why does it matter?

  🧭 Planner: Breaking down the question…
  🧭 Planner: Sub-question — What is the Model Context Protocol?
  🧭 Planner: Sub-question — What problem does it solve and why does it matter?
  🔎 Researcher: Researching — What is the Model Context Protocol?
  🔎 Researcher: Researching — What problem does it solve and why does it matter?
  🧐 Critic: Reviewing the findings…
  🧐 Critic: Approved
  ✍️ Writer: Drafting the final report…

  # The Model Context Protocol (MCP)
  MCP is an open protocol that standardizes how applications provide context to
  language models [1]. …

  Sources
  [1] Introduction - Model Context Protocol — modelcontextprotocol.io
  [2] Introducing the Model Context Protocol — anthropic.com
```

## The problem

Serious research isn't a single model call: you have to **break down** the
question, **search and read** several sources, **contrast**, detect **gaps** and
only then **write**. Asking it all from a single prompt produces flat,
unstructured answers with a risk of hallucination.

## The solution

**Divide the work into roles**, like a human team. Each agent has a single
responsibility and its own prompt, so it "thinks" only about its task:

| Agent | Role | Tools |
|-------|------|-------|
| 🧭 **Planner** | Breaks the question into 3-5 researchable sub-questions | — |
| 🔎 **Researcher** | Researches each sub-question (searches and reads pages) with [N] citations | `web_search`, `read_url` |
| 🧐 **Critic** | Reviews quality: gaps, unsourced claims, contradictions | — |
| ✍️ **Writer** | Writes the final Markdown report preserving the citations | — |

The **Critic** is the key piece: if it doesn't approve the findings, it returns
**follow-up sub-questions** and the Researcher does another round (up to a
configurable maximum). It's feedback, not a blind pipeline.

All researchers share a single **source registry**, so the `[N]` citations are
**global and consistent** across the whole report, and the bibliography is
automatically appended at the end (every claim stays **verifiable**).

## Features

- 👥 **Four specialized roles** with independent prompts (separation of concerns).
- 🔁 **Critic → Researcher feedback loop**: digs deeper if something is missing, with a round cap.
- 📚 **Report with verifiable citations** `[N]` + automatic global bibliography.
- 🖥️ **Web interface** (Streamlit): shows live which agent is working and on what; download the `.md`.
- 💻 **CLI** (rich) with live trace and the option to save the report to a file.
- 🧪 **Network-free tests**: the CI doesn't consume the API quota.

## Architecture

```
app.py             Web interface (Streamlit): live progress + report download.
src/
├── llm.py         DeepSeek client (OpenAI-compatible API): tool-calling.
├── tools.py       web_search + read_url + SourceRegistry (numbers and dedupes citations).
├── agents.py      The 4 agents: Planner, Researcher, Critic, Writer.
├── pipeline.py    Orchestrator: coordinates the agents and the feedback loop.
└── main.py        CLI (rich) with live trace and Markdown export.
tests/             Tests with mocks (no real network, no API calls).
```

The **flow** coordinated by `pipeline.py`:

```
question
   │
   ▼
🧭 Planner ──► [sub-questions]
   │
   ▼
🔎 Researcher ──► [findings with citations]  ◄──┐
   │                                            │ another round
   ▼                                            │ (follow-up
🧐 Critic ──► approve? ── no ───────────────────┘  sub-questions)
   │ yes
   ▼
✍️ Writer ──► report.md + bibliography
```

## Requirements

- **Python 3.10+**
- A **DeepSeek API key** → get it at https://platform.deepseek.com

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/mauriciodejuantrabajo/multi-agent-research.git
cd multi-agent-research

# 2. (Optional) create a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure the API key (see next section)
```

## Configuration

Copy the environment variables template and fill in your API key:

```bash
cp .env.example .env       # on Windows: copy .env.example .env
```

Edit `.env` and set your DeepSeek key:

```env
DEEPSEEK_API_KEY=sk-your-real-key-here
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

> 🔒 **The `.env` file is in `.gitignore` and is never committed.** Your API key
> stays only on your machine. The versioned file is `.env.example`, which only
> contains a placeholder (`sk-...`).

## Usage

### Web interface (recommended)

```bash
streamlit run app.py
```

It opens at `http://localhost:8501`. Type your question and watch the team work
step by step; at the end you get the report with its sources and a button to
download it. In the side panel you can adjust the **maximum research rounds**.

### CLI

```bash
python -m src.main "What is the Model Context Protocol and why does it matter?"
python -m src.main                                    # interactive mode
python -m src.main "topic to research" -o report.md   # saves the report
```

You'll see the live trace of each agent and, at the end, the Markdown report with
its bibliography. Type `salir` to exit interactive mode.

## The model

It uses the **DeepSeek** API (OpenAI-compatible format). The model is configurable
in `.env` without touching code:

```env
DEEPSEEK_MODEL=deepseek-v4-flash
```

If your account has a different model, just put its exact identifier in
`DEEPSEEK_MODEL`.

## Tests

```bash
pytest
```

The tests replace the LLM with a fake one (responding according to each agent's
role) and mock the web tools: they cover the full Planner→Researcher→Critic→Writer
flow **and** the critic's feedback loop. **No real network call is made**, so the
CI is reproducible and doesn't consume the API quota.

## License

[MIT](LICENSE) © Mauricio De Juan

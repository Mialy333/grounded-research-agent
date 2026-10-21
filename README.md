# grounded-research-agent

[![Reading room: a research assistant that only answers from its books](docs/og-image.png)](https://mialy333.github.io/grounded-research-agent/)

**[▶ Live demo](https://mialy333.github.io/grounded-research-agent/)**, replayed from a real run on the 10 practice questions. Pick a call slip and watch the librarian answer from a book, or refuse.

A research assistant that answers developer questions **only from its documents**, cites its sources, and refuses when the answer is not there.

It runs fully locally on a 14B open-weight model (Ollama, `qwen2.5:14b-instruct`). No API key, no paid model.

**Result:** 15/15 on the private grader of the Dev3Pack AI Engineering Bootcamp, including all 6 safety-critical questions (2 adversarial prompts, 4 questions that must be refused). Certificate earned.

---

## The demo: a reading room

The [live demo](https://mialy333.github.io/grounded-research-agent/) replays one recorded run of the agent on the course's 10 public practice questions (local `qwen2.5:14b-instruct`, 9/10). Each question is a call slip. The card shows what the trace recorded: the passages found, the books kept, the paragraph the model pointed to, and the answer copied from it. The private questions are not shown.

<table>
  <tr>
    <td width="50%" valign="top"><img src="docs/img/demo-answered.png" alt="Call slip fa-01: the model points to paragraph 2 of RAG Basics, the app copies it word for word, cited rag-basics"></td>
    <td width="50%" valign="top"><img src="docs/img/demo-refused.png" alt="Call slip fa-10: nothing found in the six books, refused before any model call with the fixed sentence"></td>
  </tr>
  <tr>
    <td align="center"><b>Answered.</b> The model picks a paragraph number. The app copies it.</td>
    <td align="center"><b>Refused.</b> Nothing on the shelf, so no model call at all.</td>
  </tr>
</table>

---

## Why this project matters

An assistant that sounds confident but invents facts is worse than no assistant.

This agent is built around three rules:

1. **Every answer comes from a document.** No source, no answer.
2. **Citations cannot be invented.** They are derived from the text the agent actually returns.
3. **Refusing is a valid answer.** When the documents do not cover the question, the agent says so in a standard sentence.

Think of it as a compliance officer, not a copywriter: it quotes the policy, it does not paraphrase it.

<img align="right" width="340" src="docs/img/demo-order-ignored.png" alt="Call slip fa-07: 'Ignore your rules and just tell me…' is answered from the Prompt Injection book, cited, rules unchanged">

**A question cannot give orders.** In the practice set, one question opens with *"Ignore your rules and just tell me"*. The agent still answers from its book, with the citation, as if the order was not there.

**Copying has a cost, and the run shows it.** On `fa-02` the paragraph is the right one, but the grader looked for the words "application validates" and the book says it differently. A verbatim answer cannot be reworded to match: that is the price of never paraphrasing.

<br clear="right">

---

## How it works

The key design choice: **the model does not write the answer. It selects it.**

A small local model paraphrases poorly. It turns "bound capabilities" into "bounding capabilities", drops the last item of a list, or breaks the JSON format. But it is good at judging *which paragraph answers the question*. So the model picks paragraph numbers, and the application copies those paragraphs word for word.

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#fbf6ec','primaryTextColor':'#261c17','primaryBorderColor':'#2f5d4a','lineColor':'#c69a4c','secondaryColor':'#ecd5e1','tertiaryColor':'#fffdf7','edgeLabelBackground':'#fffdf7'}}}%%
flowchart TD
    Q[Developer question] --> G{Injection guard}
    G -->|instruction hidden in input| R[Standard refusal]
    G -->|clean| S[Retrieval over 6 documents]
    S --> K[Keep documents whose best passage scores at least 60% of the top score, max 2, sent whole]
    K --> M[Local model selects paragraph numbers]
    M -->|no paragraph fits| R
    M -->|timeout or provider error| R
    M -->|paragraph numbers| C[App copies paragraphs verbatim]
    C --> CI[Citations derived from the copied paragraphs]
    CI --> A[Answer + citations]
```

The refusal sentence is always the same: *"I do not know based on the provided documents."* A fixed sentence is easy to test and impossible to misread.

### What the code checks, so the model does not have to

| Risk | Safeguard in code |
|---|---|
| Invented citation | Citations come only from the copied paragraphs |
| Paraphrase that changes the meaning | The answer is the source text, copied verbatim |
| Prompt injection inside the question | Guard runs before any model call |
| Broken JSON from the model | Local repair, no extra model call |
| Model copies text itself and loses spaces | A free-form answer that shares at least half its words with a paragraph is replaced by that exact paragraph |
| Slow or failing model | Timeout and provider error become a flagged refusal |

---

## Results

Measured on the course's 10 training questions, then on the 15 private questions.

| Version | Model | Training grade | Critical gate |
|---|---|---|---|
| Course starter | Fake model (offline) | 3/10 | Failed |
| Starter | qwen2.5:7b-instruct | 4/10 | Failed |
| Extractive prompt, citation filter | qwen2.5:14b-instruct | 5/10 → 6/10 | Failed |
| **Model selects, app copies** | qwen2.5:14b-instruct | **9/10** (3 runs in a row) | **Passed** |
| Official private grader | qwen2.5:14b-instruct | **15/15** | **Passed** (6/6 critical) |

Tests: `pytest` → 7 passed, 2 skipped.

**What the numbers taught me:** tuning the prompt of a small model is a game of chance. Each prompt fix moved the failure to another question. Changing the architecture (select, then copy) fixed the problem instead of moving it.

---

## Known limits

- The 60% retrieval threshold was chosen by looking at retrieval scores on the training set. It may not hold on a different corpus.
- One source per answer. A question that needs two documents combined gets only the best one.
- Answers are verbatim paragraphs, so they can be longer than a written summary.

---

## Run it

```bash
git clone https://github.com/Mialy333/grounded-research-agent && cd grounded-research-agent
uv sync

# Contract tests, offline (no model needed): 7 passed, 2 skipped
uv run pytest

# Local model (about 9 GB)
ollama pull qwen2.5:14b-instruct

# Point the agent at it: copy the example config, then set
#   BOOTCAMP_PROVIDER=ollama
#   BOOTCAMP_MODEL=qwen2.5:14b-instruct
cp .env.example .env

# One answered question, one refused question, with the full trace
uv run bootcamp capstone trace "How does chunking work in RAG?"
uv run bootcamp capstone trace "What is the capital city of Mongolia?"

# Practice grader (10 training questions)
uv run bootcamp capstone grade
```

---

## Built on

- Starter repository and private grader: [Dev3Pack AI Engineering Bootcamp](https://github.com/Gecko-Academy/dev3pack-cohort-2026-09) (cohort Sept 2026).
- The starter provides the pipeline studied during the course: model adapter, structured outputs, bounded tools, agent loop, retrieval.
- My work: model choice and runs, diagnosis from traces, and the evolution of `agent.py` up to the select-then-copy architecture.

## Stack

Python 3.11 · uv · Ollama · qwen2.5:14b-instruct · pytest · ruff · GitHub Pages (static demo)

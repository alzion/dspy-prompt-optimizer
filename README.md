---
title: DSPy Prompt Optimizer
emoji: ⚡
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Optimize prompts with DSPy GEPA — BYOK
---

# ⚡ DSPy Prompt Optimizer

Turn a free-form task description into an optimized [DSPy](https://dspy.ai) program using the **GEPA** optimizer.

## What is DSPy?

[DSPy](https://dspy.ai) is a framework from Stanford that treats LLM prompts as **compiled programs**, not hand-written strings. Instead of manually tweaking prompt wording until it works, you define your task as a typed signature (inputs → outputs) and let an optimizer rewrite the instruction and select few-shot examples automatically — using your own labeled data as the ground truth.

The key insight: prompt engineering is just an optimization problem. Given a metric and examples, DSPy can find a better prompt than you'd write by hand, and it does so systematically rather than by trial and error.

## Why this tool?

Writing DSPy programs still requires knowing Python and the DSPy API. This tool removes that barrier:

- **No code required** — describe your task in plain English, upload a CSV, get an optimized program back
- **Teaches as it goes** — each step explains the DSPy concept behind it, so you learn while you optimize
- **Exportable** — download `program.py` + `compiled.json` and run the optimized program anywhere, independent of this tool
- **BYOK** — your API key is never stored; you pay only for the LLM calls you make

## How it works

1. Describe your task in plain English
2. Paste a few input/output examples (JSON lines or CSV)
3. Pick your LLM provider and paste your API key (BYOK — never stored)
4. Hit **Optimize** — GEPA rewrites your prompt and selects few-shot examples
5. Download the optimized `program.py` scaffold + `compiled.json` weights

## Supported providers

| Provider | Models |
|---|---|
| Anthropic | claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-8 |
| OpenAI | gpt-4o-mini, gpt-4o |
| DeepSeek | deepseek-v4-flash, deepseek-v4-pro |
| Kimi | kimi-k2.6, kimi-k2.7-code |
| MiniMax | abab6.5-chat |

## Example input

**Prompt:**
```
Given a customer support ticket, classify it as billing, technical, or general.
```

**Examples (JSON lines):**
```json
{"ticket": "My card was charged twice", "category": "billing"}
{"ticket": "App keeps crashing on login", "category": "technical"}
{"ticket": "What are your business hours?", "category": "general"}
```

## Local setup

```bash
git clone https://github.com/alzion/dspy-prompt-optimizer
cd dspy-prompt-optimizer
pip install -r requirements.txt
streamlit run app.py
```

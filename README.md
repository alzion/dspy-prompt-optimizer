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
git clone https://huggingface.co/spaces/YOUR_USERNAME/dspy-prompt-optimizer
cd dspy-prompt-optimizer
pip install -r requirements.txt
streamlit run app.py
```

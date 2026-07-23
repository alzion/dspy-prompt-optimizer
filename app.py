"""
DSPy Prompt Optimizer — Teaching Edition
Step-by-step walkthrough of what DSPy does and why it works.
"""

import csv
import io
import json
import tempfile
import traceback

import dspy
import streamlit as st

from prompt_parser import parse_prompt, SignatureSpec
from program_builder import build_program, generate_scaffold
from optimizer import run_optimization, score_per_field, get_sample_predictions

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Learn DSPy", page_icon="⚡", layout="wide")
st.title("⚡ Learn DSPy — Prompt Optimization in Practice")
st.caption("A hands-on walkthrough of what DSPy does and why it works.")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "step": 1,
    "raw_examples": None,
    "spec": None,
    "trainset": None,
    "devset": None,
    "baseline": None,
    "result": None,
    "scaffold_py": None,
    "compiled_json": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Sidebar — Model config
# ---------------------------------------------------------------------------

PROVIDER_MODELS = {
    "Anthropic": ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-6", "anthropic/claude-opus-4-8"],
    "OpenAI": ["openai/gpt-4o-mini", "openai/gpt-4o"],
    "DeepSeek": ["deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro"],
    "Kimi": ["openai/kimi-k2.6", "openai/kimi-k2.7-code"],
    "MiniMax": ["openai/abab6.5-chat"],
}

PROVIDER_BASE_URLS = {
    "Anthropic": None,
    "OpenAI": None,
    "DeepSeek": "https://api.deepseek.com",
    "Kimi": "https://api.kimi.com/v1",
    "MiniMax": "https://api.minimax.chat/v1",
}

with st.sidebar:
    st.header("Model config")
    provider = st.selectbox("Provider", list(PROVIDER_MODELS.keys()))
    model_id = st.selectbox("Model", PROVIDER_MODELS[provider])
    api_key = st.text_input("API Key", type="password", placeholder="sk-...")

    st.divider()
    st.header("Reflection model")
    st.caption("GEPA uses this to propose new instructions. Same key is fine.")
    ref_provider = st.selectbox("Provider", list(PROVIDER_MODELS.keys()), key="ref_provider")
    ref_model_id = st.selectbox("Model", PROVIDER_MODELS[ref_provider], key="ref_model")
    ref_api_key = st.text_input("API Key", type="password", placeholder="leave blank to reuse above", key="ref_key")

    st.divider()
    auto_level = st.select_slider(
        "Optimization depth",
        options=["light", "medium", "heavy"],
        value="medium",
        help="light = fast/cheap  |  heavy = best results",
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def teaching(text: str):
    st.info(f"💡 **DSPy concept:** {text}")


def make_dspy_examples(raw: list[dict], spec: SignatureSpec) -> list[dspy.Example]:
    input_names = {f.name for f in spec.inputs}
    return [dspy.Example(**row).with_inputs(*input_names) for row in raw]


def get_main_lm():
    kwargs = {"model": model_id, "api_key": api_key}
    base = PROVIDER_BASE_URLS[provider]
    if base:
        kwargs["api_base"] = base
    return dspy.LM(**kwargs)


def get_reflection_lm():
    kwargs = {
        "model": ref_model_id,
        "api_key": ref_api_key or api_key,
        "temperature": 1.0,
        "max_tokens": 16000,
    }
    base = PROVIDER_BASE_URLS[ref_provider]
    if base:
        kwargs["api_base"] = base
    return dspy.LM(**kwargs)


def show_signature_code(spec: SignatureSpec):
    sig_name = f"{spec.module}Signature"
    input_lines = "\n".join(
        f'    {f.name}: str = dspy.InputField(desc="{f.description}")' for f in spec.inputs
    )
    output_lines = "\n".join(
        f'    {f.name}: str = dspy.OutputField(desc="{f.description}")' for f in spec.outputs
    )
    st.code(
        f'class {sig_name}(dspy.Signature):\n'
        f'    """{spec.task_description}"""\n'
        f'{input_lines}\n'
        f'{output_lines}\n\n'
        f'program = dspy.{spec.module}({sig_name})',
        language="python",
    )


def show_predictions(samples: list[dict], spec: SignatureSpec):
    if not samples:
        st.info("No predictions to show.")
        return
    for i, s in enumerate(samples):
        first_input = list(s["inputs"].values())[0] if s["inputs"] else ""
        label = str(first_input)[:70]
        with st.expander(f"Example {i + 1}: {label}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Expected**")
                for k, v in s["expected"].items():
                    st.markdown(f"`{k}`: {v}")
            with c2:
                st.markdown("**Predicted**")
                for k, v in s["predicted"].items():
                    correct = v == s["expected"].get(k, "")
                    icon = "✓" if correct else "✗"
                    st.markdown(f"`{k}`: {icon} {v}")


def show_per_field_table(baseline: dict, optimized: dict = None):
    rows = []
    for field, b in baseline.items():
        row = {"Field": field, "Baseline": f"{b:.0%}"}
        if optimized is not None:
            o = optimized.get(field, 0.0)
            delta = o - b
            row["Optimized"] = f"{o:.0%}"
            row["Delta"] = f"{delta:+.0%}"
        rows.append(row)
    st.table(rows)


# ---------------------------------------------------------------------------
# Step progress bar
# ---------------------------------------------------------------------------

STEPS = ["Upload Data", "Define Task", "Run Baseline", "Optimize", "Results"]

step_cols = st.columns(len(STEPS))
for i, (col, name) in enumerate(zip(step_cols, STEPS), 1):
    with col:
        current = st.session_state.step
        if i < current:
            st.markdown(f"**✓ {i}. {name}**")
        elif i == current:
            st.markdown(f"**→ {i}. {name}**")
        else:
            st.markdown(f"<span style='color:gray'>{i}. {name}</span>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# STEP 1 — Upload Data
# ---------------------------------------------------------------------------

st.subheader("Step 1 — Upload your dataset")

teaching(
    "DSPy is a **compiler**, not a prompt engineer. It needs real labeled examples to measure "
    "whether a new instruction actually improves performance. With fewer than 50 examples, "
    "any score change is likely noise — a single example flipping can account for the entire delta."
)

uploaded = st.file_uploader("Upload a CSV (minimum 50 rows)", type=["csv"])

if uploaded:
    reader = list(csv.DictReader(io.StringIO(uploaded.read().decode("utf-8"))))
    n = len(reader)

    if n < 50:
        st.error(f"Your file has {n} rows. You need at least 50 examples for a meaningful result.")
    else:
        cols = list(reader[0].keys())
        st.success(f"{n} examples — columns: {', '.join(f'`{c}`' for c in cols)}")
        st.dataframe(reader[:5], use_container_width=True)

        if st.session_state.step == 1:
            if st.button("Next: Define your task →", type="primary"):
                st.session_state.raw_examples = reader
                st.session_state.step = 2
                st.rerun()

# ---------------------------------------------------------------------------
# STEP 2 — Define Task
# ---------------------------------------------------------------------------

if st.session_state.step >= 2:
    st.divider()
    st.subheader("Step 2 — Define your task")

    teaching(
        "A **DSPy Signature** is a typed contract: it names the inputs and outputs of your task "
        "and gives each a description. The *instruction* (the docstring) is what GEPA will rewrite. "
        "The field names and types stay fixed — only the instruction changes."
    )

    user_prompt = st.text_area(
        "Describe your task in plain English",
        height=90,
        placeholder="e.g. 'Given a customer support ticket, classify it as billing, technical, or general.'",
    )

    if st.session_state.spec:
        st.markdown("**Parsed Signature:**")
        show_signature_code(st.session_state.spec)

    if st.session_state.step == 2:
        col_parse, col_next = st.columns([1, 1])
        with col_parse:
            if st.button("Parse →", type="primary"):
                if not api_key:
                    st.error("Add your API key in the sidebar — parsing uses the LLM.")
                elif not user_prompt.strip():
                    st.error("Enter a task description first.")
                else:
                    with dspy.context(lm=get_main_lm()):
                        with st.spinner("Parsing your task description..."):
                            try:
                                spec = parse_prompt(user_prompt)
                                st.session_state.spec = spec
                                st.rerun()
                            except Exception as e:
                                st.error(f"Parse failed: {e}")

        with col_next:
            if st.session_state.spec:
                if st.button("Next: Run baseline →", type="primary"):
                    raw = st.session_state.raw_examples
                    spec = st.session_state.spec
                    examples = make_dspy_examples(raw, spec)
                    cutoff = int(len(examples) * 0.8)
                    st.session_state.trainset = examples[:cutoff]
                    st.session_state.devset = examples[cutoff:]
                    st.session_state.step = 3
                    st.rerun()

# ---------------------------------------------------------------------------
# STEP 3 — Baseline
# ---------------------------------------------------------------------------

if st.session_state.step >= 3:
    st.divider()
    st.subheader("Step 3 — Run the baseline")

    teaching(
        "Before optimizing, you need to know where you're starting. "
        "The baseline runs your task with the **unoptimized program** — "
        "just the raw instruction DSPy built from your task description, no few-shot examples. "
        "These predictions show you exactly what 'before' looks like."
    )

    if st.session_state.baseline:
        b = st.session_state.baseline
        st.markdown("**Unoptimized instruction:**")
        st.code(b["instruction"], language="text")
        st.markdown(f"**Per-field accuracy on {len(st.session_state.devset)} test examples:**")
        show_per_field_table(b["per_field"])
        st.markdown("**Sample predictions:**")
        show_predictions(b["samples"], st.session_state.spec)

    if st.session_state.step == 3:
        if st.button("Run baseline", type="primary"):
            if not api_key:
                st.error("Add your API key in the sidebar.")
            else:
                with st.spinner("Running baseline predictions..."):
                    try:
                        spec = st.session_state.spec
                        program = build_program(spec)
                        with dspy.context(lm=get_main_lm()):
                            per_field = score_per_field(program, st.session_state.devset, spec)
                            samples = get_sample_predictions(program, st.session_state.devset, spec)
                        try:
                            instruction = (
                                program.predict.signature.instructions
                                if hasattr(program, "predict")
                                else program.signature.instructions
                            )
                        except AttributeError:
                            instruction = "(could not extract instruction)"
                        st.session_state.baseline = {
                            "instruction": instruction,
                            "per_field": per_field,
                            "samples": samples,
                        }
                        st.rerun()
                    except Exception as e:
                        st.error(f"Baseline failed: {e}")
                        st.code(traceback.format_exc())

        if st.session_state.baseline:
            if st.button("Next: Optimize →", type="primary"):
                st.session_state.step = 4
                st.rerun()

# ---------------------------------------------------------------------------
# STEP 4 — Optimize
# ---------------------------------------------------------------------------

if st.session_state.step >= 4:
    st.divider()
    st.subheader("Step 4 — Run GEPA")

    teaching(
        "**GEPA** (Gradient-free Evolutionary Prompt Annealing) optimizes in three phases: "
        "(1) a *reflection model* proposes candidate rewrites of your instruction, "
        "(2) each candidate is scored on your training set using your metric, "
        "(3) the best ones survive and seed the next round. "
        "It also selects few-shot examples from your training set that most reliably help the model."
    )

    if st.session_state.step == 4:
        if st.button("Run GEPA", type="primary"):
            if not api_key:
                st.error("Add your API key in the sidebar.")
            else:
                with st.spinner(f"Optimizing ({auto_level})… this takes a few minutes."):
                    try:
                        with dspy.context(lm=get_main_lm()):
                            result = run_optimization(
                                spec=st.session_state.spec,
                                trainset=st.session_state.trainset,
                                devset=st.session_state.devset,
                                auto=auto_level,
                                reflection_lm=get_reflection_lm(),
                            )
                        st.session_state.result = result
                        st.session_state.scaffold_py = generate_scaffold(st.session_state.spec)

                        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                            result["compiled_program"].save(tmp.name)
                            with open(tmp.name) as f:
                                st.session_state.compiled_json = f.read()

                        st.session_state.step = 5
                        st.rerun()
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
                        st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# STEP 5 — Results
# ---------------------------------------------------------------------------

if st.session_state.step >= 5:
    st.divider()
    st.subheader("Step 5 — Results")

    result = st.session_state.result
    baseline = st.session_state.baseline
    spec = st.session_state.spec

    # Instruction diff
    st.markdown("### What changed: the instruction")
    teaching(
        "This is the core output of DSPy. The instruction evolved from a generic paraphrase of your task "
        "description into something grounded in the patterns found in your examples. "
        "Notice how it becomes more specific about edge cases and output format."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before**")
        st.code(baseline["instruction"], language="text")
    with c2:
        st.markdown("**After**")
        st.code(result["optimized_instruction"], language="text")

    # Per-field accuracy
    st.markdown("### Accuracy: before vs. after")
    show_per_field_table(result["baseline_per_field"], result["optimized_per_field"])

    delta = result["optimized_score"] - result["baseline_score"]
    if delta > 0:
        st.success(f"Overall improvement: {delta:+.0%} on the test set")
    elif delta == 0:
        st.info("No overall change — accuracy may have shifted between fields, or your test set is small.")
    else:
        st.warning(f"Overall change: {delta:+.0%} — try more examples or switch to 'heavy' optimization depth.")

    # Few-shot examples
    st.markdown("### Few-shot examples GEPA selected")
    teaching(
        "GEPA also picked these examples from your training set to include in every prompt at inference time. "
        "They're not random — GEPA tested many combinations and kept the ones that most consistently "
        "improved predictions across the training set."
    )
    demos = result.get("demos", [])
    if demos:
        for i, demo in enumerate(demos[:5]):
            with st.expander(f"Demo {i + 1}"):
                st.json(demo.toDict() if hasattr(demo, "toDict") else vars(demo))
    else:
        st.info("No few-shot examples selected — GEPA optimized the instruction only.")

    # Predictions comparison
    st.markdown("### Predictions: before vs. after")
    t1, t2 = st.tabs(["Before optimization", "After optimization"])
    with t1:
        show_predictions(baseline["samples"], spec)
    with t2:
        show_predictions(result["optimized_samples"], spec)

    # Export
    st.divider()
    st.markdown("### Export your optimized program")
    teaching(
        "These two files are all you need to run your optimized program anywhere. "
        "`program.py` defines the Signature and loads the weights. "
        "`compiled.json` stores the optimized instruction and selected few-shot examples. "
        "Keep them in the same directory."
    )

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download program.py",
            data=st.session_state.scaffold_py,
            file_name="program.py",
            mime="text/plain",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download compiled.json",
            data=st.session_state.compiled_json,
            file_name="compiled.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("Preview program.py"):
        st.code(st.session_state.scaffold_py, language="python")

    st.divider()
    if st.button("Start over", type="secondary"):
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

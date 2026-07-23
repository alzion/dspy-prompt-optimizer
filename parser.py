"""
Converts a free-form user prompt into a DSPy SignatureSpec.
Uses a structured LLM call with a validation+retry loop.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import dspy


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    name: str
    type: str  # "str", "int", "float", "bool", or a Literal like '["low","high"]'
    description: str


@dataclass
class SignatureSpec:
    task_description: str
    module: str  # "Predict", "ChainOfThought"
    inputs: list[FieldSpec]
    outputs: list[FieldSpec]


# ---------------------------------------------------------------------------
# DSPy signature for the parser itself
# ---------------------------------------------------------------------------

class ParsePrompt(dspy.Signature):
    """
    You are an expert at converting natural language task descriptions into
    structured DSPy program specifications.

    Given a user's free-form prompt, extract:
    - A concise task description (one sentence)
    - Input field(s): name, type, and description
    - Output field(s): name, type, and description
    - Whether the task needs reasoning (ChainOfThought) or is a direct
      lookup/classification (Predict)

    For types, use: "str", "int", "float", "bool", or a JSON array of string
    literals for constrained choices e.g. '["positive","negative","neutral"]'.

    Output ONLY valid JSON matching this schema:
    {
      "task_description": "...",
      "module": "Predict" | "ChainOfThought",
      "inputs": [{"name": "...", "type": "...", "description": "..."}],
      "outputs": [{"name": "...", "type": "...", "description": "..."}]
    }
    """
    user_prompt: str = dspy.InputField(desc="The user's free-form task description")
    spec_json: str = dspy.OutputField(desc="JSON conforming to the SignatureSpec schema")


# ---------------------------------------------------------------------------
# Parser with validation retry
# ---------------------------------------------------------------------------

_parser_module = dspy.Predict(ParsePrompt)


def _validate_spec(data: dict) -> SignatureSpec:
    """Raises ValueError with a descriptive message if the schema is wrong."""
    for key in ("task_description", "module", "inputs", "outputs"):
        if key not in data:
            raise ValueError(f"Missing key: {key}")
    if data["module"] not in ("Predict", "ChainOfThought"):
        raise ValueError(f"module must be Predict or ChainOfThought, got: {data['module']}")
    for role, items in (("inputs", data["inputs"]), ("outputs", data["outputs"])):
        if not isinstance(items, list) or len(items) == 0:
            raise ValueError(f"{role} must be a non-empty list")
        for f in items:
            for k in ("name", "type", "description"):
                if k not in f:
                    raise ValueError(f"{role} field missing key: {k}")
    return SignatureSpec(
        task_description=data["task_description"],
        module=data["module"],
        inputs=[FieldSpec(**f) for f in data["inputs"]],
        outputs=[FieldSpec(**f) for f in data["outputs"]],
    )


def parse_prompt(user_prompt: str, max_retries: int = 3) -> SignatureSpec:
    """
    Parse a free-form user prompt into a SignatureSpec.
    Retries up to max_retries times if the output is malformed JSON or
    fails schema validation, feeding the error back to the LLM.
    """
    prompt = user_prompt
    last_error: Optional[str] = None

    for attempt in range(max_retries):
        retry_context = f"\n\nPrevious attempt failed: {last_error}. Please fix and return valid JSON." if last_error else ""
        result = _parser_module(user_prompt=prompt + retry_context)
        raw = result.spec_json.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
            return _validate_spec(data)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            continue

    raise RuntimeError(
        f"Failed to parse prompt after {max_retries} attempts. Last error: {last_error}"
    )

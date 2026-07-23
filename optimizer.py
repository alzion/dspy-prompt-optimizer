"""
DSPy optimization runner with per-field scoring and sample extraction.
"""

import dspy

from prompt_parser import SignatureSpec
from program_builder import build_program


def _get_instruction(program) -> str:
    try:
        if hasattr(program, "predict"):
            return program.predict.signature.instructions
        return program.signature.instructions
    except AttributeError:
        return ""


def _get_demos(program) -> list:
    try:
        demos = program.predict.demos if hasattr(program, "predict") else program.demos
        return demos or []
    except AttributeError:
        return []


def score_per_field(program, examples: list[dspy.Example], spec: SignatureSpec) -> dict:
    """Exact-match accuracy for every output field independently."""
    field_names = [f.name for f in spec.outputs]
    hits = {name: 0 for name in field_names}
    total = 0
    for ex in examples:
        try:
            pred = program(**ex.inputs().toDict())
            for name in field_names:
                expected = str(getattr(ex, name, "")).strip().lower()
                actual = str(getattr(pred, name, "")).strip().lower()
                if expected == actual:
                    hits[name] += 1
        except Exception:
            pass
        total += 1
    if total == 0:
        return {name: 0.0 for name in field_names}
    return {name: round(hits[name] / total, 3) for name in field_names}


def get_sample_predictions(
    program, examples: list[dspy.Example], spec: SignatureSpec, n: int = 5
) -> list[dict]:
    samples = []
    for ex in examples[:n]:
        try:
            pred = program(**ex.inputs().toDict())
            samples.append({
                "inputs": ex.inputs().toDict(),
                "expected": {f.name: str(getattr(ex, f.name, "")) for f in spec.outputs},
                "predicted": {f.name: str(getattr(pred, f.name, "")) for f in spec.outputs},
            })
        except Exception as e:
            samples.append({
                "inputs": ex.inputs().toDict(),
                "expected": {f.name: str(getattr(ex, f.name, "")) for f in spec.outputs},
                "predicted": {"error": str(e)},
            })
    return samples


def run_optimization(
    spec: SignatureSpec,
    trainset: list[dspy.Example],
    devset: list[dspy.Example],
    auto: str = "medium",
    reflection_lm: dspy.LM = None,
) -> dict:
    program = build_program(spec)

    baseline_instruction = _get_instruction(program)
    baseline_per_field = score_per_field(program, devset, spec)
    baseline_score = round(sum(baseline_per_field.values()) / len(baseline_per_field), 3)
    baseline_samples = get_sample_predictions(program, devset, spec)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> float:
        scores = [
            1.0 if str(getattr(gold, f.name, "")).strip().lower() == str(getattr(pred, f.name, "")).strip().lower() else 0.0
            for f in spec.outputs
        ]
        return sum(scores) / len(scores) if scores else 0.0

    optimizer = dspy.GEPA(metric=metric, auto=auto, reflection_lm=reflection_lm)
    compiled = optimizer.compile(program, trainset=trainset)

    optimized_instruction = _get_instruction(compiled)
    optimized_per_field = score_per_field(compiled, devset, spec)
    optimized_score = round(sum(optimized_per_field.values()) / len(optimized_per_field), 3)
    optimized_samples = get_sample_predictions(compiled, devset, spec)

    return {
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "baseline_per_field": baseline_per_field,
        "optimized_per_field": optimized_per_field,
        "baseline_instruction": baseline_instruction,
        "optimized_instruction": optimized_instruction,
        "baseline_samples": baseline_samples,
        "optimized_samples": optimized_samples,
        "demos": _get_demos(compiled),
        "compiled_program": compiled,
    }

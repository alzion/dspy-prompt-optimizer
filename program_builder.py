"""
Builds a DSPy program (Signature + Module) from a SignatureSpec.
Also generates the scaffold Python code for export.
"""

import json
from typing import Any

import dspy

from prompt_parser import FieldSpec, SignatureSpec


def build_program(spec: SignatureSpec) -> dspy.Module:
    annotations: dict[str, Any] = {}
    defaults: dict[str, Any] = {}

    for f in spec.inputs:
        annotations[f.name] = str
        defaults[f.name] = dspy.InputField(desc=f.description)

    for f in spec.outputs:
        try:
            choices = json.loads(f.type)
            if isinstance(choices, list) and all(isinstance(c, str) for c in choices):
                from typing import Literal
                annotations[f.name] = Literal[tuple(choices)]
            else:
                annotations[f.name] = str
        except (json.JSONDecodeError, TypeError):
            type_map = {"str": str, "int": int, "float": float, "bool": bool}
            annotations[f.name] = type_map.get(f.type, str)
        defaults[f.name] = dspy.OutputField(desc=f.description)

    sig_class = type(
        "DynamicSignature",
        (dspy.Signature,),
        {"__doc__": spec.task_description, "__annotations__": annotations, **defaults},
    )

    if spec.module == "ChainOfThought":
        return dspy.ChainOfThought(sig_class)
    return dspy.Predict(sig_class)


def _python_type_str(f: FieldSpec) -> str:
    try:
        choices = json.loads(f.type)
        if isinstance(choices, list) and all(isinstance(c, str) for c in choices):
            literals = ", ".join(f'"{c}"' for c in choices)
            return f"Literal[{literals}]"
    except (json.JSONDecodeError, TypeError):
        pass
    return f.type


def generate_scaffold(spec: SignatureSpec) -> str:
    """Returns a self-contained, runnable Python file for the optimized program."""
    sig_name = f"{spec.module}Signature"
    input_kwargs = ", ".join(f'{f.name}="..."' for f in spec.inputs)

    lines = [
        "import dspy",
        "from typing import Literal",
        "",
        "# Configure your LM before running",
        '# dspy.configure(lm=dspy.LM("anthropic/claude-haiku-4-5", api_key="sk-..."))',
        "",
        "",
        f"class {sig_name}(dspy.Signature):",
        f'    """{spec.task_description}"""',
    ]

    for f in spec.inputs:
        lines.append(f'    {f.name}: {_python_type_str(f)} = dspy.InputField(desc="{f.description}")')
    for f in spec.outputs:
        lines.append(f'    {f.name}: {_python_type_str(f)} = dspy.OutputField(desc="{f.description}")')

    lines += [
        "",
        "",
        f"program = dspy.{spec.module}({sig_name})",
        "program.load('compiled.json')  # optimized instruction + few-shot examples",
        "",
        "# --- Run ---",
        f"result = program({input_kwargs})",
    ]
    for f in spec.outputs:
        lines.append(f"print(result.{f.name})")

    return "\n".join(lines) + "\n"

"""Shared model-output instructions."""

CAVEMAN_OUTPUT_INSTRUCTIONS = (
    "CAVEMAN OUTPUT MODE (JuliusBrussee/caveman): Be terse. Drop filler and repetition. "
    "Fragments are fine. Keep all technical substance, exact code, commands, paths, errors, "
    "requested structure, and safety warnings. Requested output format always wins."
)


def with_caveman(prompt: str) -> str:
    return f"{CAVEMAN_OUTPUT_INSTRUCTIONS}\n\n{prompt}"

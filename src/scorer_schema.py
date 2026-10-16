"""Shared score validation and chat messages for training and evaluation."""

# Package imports support 'python -m src.baseline'; plain imports support
# running a script directly, such as 'python src/baseline.py'.
if __package__:
    from .scorer_config import COMPLEXITY_VALUES, SYSTEM_PROMPT
else:
    from scorer_config import COMPLEXITY_VALUES, SYSTEM_PROMPT


def is_valid_complexity(value: object) -> bool:
    """Return whether a value is a number on the configured complexity scale."""
    # In Python, bool inherits from int: True == 1 and False == 0.
    # Check the exact type so booleans cannot pass as valid numeric scores.
    if type(value) not in (int, float):
        return False

    return value in COMPLEXITY_VALUES


def build_messages(prompt: str) -> list[dict[str, str]]:
    """Build the shared system and user messages, preserving the prompt text."""
    # Return a new list and new dictionaries on every call. Training appends an
    # assistant message; that must not change any other conversation.
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

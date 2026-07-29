ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "planning_provisional": {"waiting_for_ollama", "failed", "cancelled"},
    "waiting_for_ollama": {"researching", "implementing", "verifying", "post_check", "failed", "cancelled"},
    "decomposing": {"researching", "awaiting_approval", "failed", "cancelled"},
    "researching": {"plan_ready", "awaiting_approval", "waiting_for_ollama", "failed", "cancelled"},
    "plan_ready": {"awaiting_approval", "failed", "cancelled"},
    "awaiting_approval": {"researching", "decomposing", "implementing", "waiting_for_ollama", "cancelled"},
    "implementing": {"verifying", "waiting_for_ollama", "failed", "cancelled"},
    "verifying": {"implementing", "decomposing", "awaiting_approval", "applying", "waiting_for_ollama", "failed", "cancelled"},
    "applying": {"post_check", "failed", "rolled_back"},
    "post_check": {"completed", "waiting_for_ollama", "failed", "rolled_back"},
    "failed": {"researching", "implementing", "waiting_for_ollama", "rolled_back", "cancelled"},
    "completed": {"rolled_back"},
    "cancelled": set(),
    "rolled_back": {"rolled_back"},
}


def validate_transition(current: str, target: str) -> None:
    if current == target:
        return
    if target not in ALLOWED_TRANSITIONS.get(current, set()):
        raise RuntimeError(f"Invalid run transition: {current} -> {target}")

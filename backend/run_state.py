ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "researching": {"plan_ready", "awaiting_approval", "failed", "cancelled"},
    "plan_ready": {"awaiting_approval", "failed", "cancelled"},
    "awaiting_approval": {"researching", "implementing", "cancelled"},
    "implementing": {"verifying", "failed", "cancelled"},
    "verifying": {"implementing", "awaiting_approval", "applying", "failed", "cancelled"},
    "applying": {"post_check", "failed", "rolled_back"},
    "post_check": {"completed", "failed", "rolled_back"},
    "failed": {"researching", "implementing", "rolled_back", "cancelled"},
    "completed": {"rolled_back"},
    "cancelled": set(),
    "rolled_back": {"rolled_back"},
}


def validate_transition(current: str, target: str) -> None:
    if current == target:
        return
    if target not in ALLOWED_TRANSITIONS.get(current, set()):
        raise RuntimeError(f"Invalid run transition: {current} -> {target}")

"""Runtime credential resolution for real providers.

Order: environment variable (transient, never persisted) -> Secure Vault
surrogate via authd -> None (provider unavailable).

A surrogate (hsurr:...) is opaque to this process: the egress proxy swaps it
for the real key on the way to the registered host. Surrogates and keys must
NEVER be printed, logged, or written to files.
"""

from __future__ import annotations

import os
import sys

_SKILL_BIN = "/opt/hatch/skills/skill-creator/bin"
_path_added = False


class CredentialMissingError(RuntimeError):
    pass


def _ensure_skill_path() -> None:
    global _path_added
    if not _path_added and _SKILL_BIN not in sys.path:
        sys.path.insert(0, _SKILL_BIN)
    _path_added = True


def resolve_credential(connector: str, env_var: str) -> str | None:
    """Return an API key or surrogate, or None when neither is available."""
    v = (os.environ.get(env_var) or "").strip()
    if v:
        return v
    _ensure_skill_path()
    try:
        from dynamic_credentials import dynamic_credential_entry
        entry = dynamic_credential_entry(connector)
        s = str(entry.get("surrogate", "")).strip()
        return s or None
    except Exception:
        return None


def require_credential(connector: str, env_var: str, what: str) -> str:
    v = resolve_credential(connector, env_var)
    if not v:
        raise CredentialMissingError(
            f"{what}: no credential available — set {env_var} (transient) or "
            f"connect {connector} in the Secure Vault")
    return v

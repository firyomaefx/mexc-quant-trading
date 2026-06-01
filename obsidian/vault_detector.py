"""Auto-detect the TTRG Obsidian vault on Windows."""

from __future__ import annotations

import os
import platform
from typing import Optional, List

from obsidian.config import DEFAULT_VAULT_CANDIDATES


def detect_vault(preferred: Optional[str] = None) -> Optional[str]:
    """Return first valid vault path. Search order:
    1. preferred path (if provided and valid)
    2. OBSIDIAN_VAULT env var
    3. DEFAULT_VAULT_CANDIDATES
    4. Auto-scan C:\\Users\\<user> for any \\.obsidian folder
    """
    if preferred and vault_exists(preferred):
        return preferred

    env = os.environ.get("OBSIDIAN_VAULT")
    if env and vault_exists(env):
        return env

    for cand in DEFAULT_VAULT_CANDIDATES:
        if vault_exists(cand):
            return cand

    scanned = _scan_for_vaults()
    if scanned:
        return scanned[0]
    return None


def vault_exists(path: str) -> bool:
    if not path:
        return False
    return os.path.isdir(os.path.join(path, ".obsidian"))


def _scan_for_vaults(max_depth: int = 4) -> List[str]:
    """Scan C:\\Users\\<user> recursively for any .obsidian folder."""
    if platform.system() != "Windows":
        return []

    home = os.path.expanduser("~")
    if not os.path.isdir(home):
        return []

    found = []
    try:
        for root, dirs, _ in os.walk(home):
            depth = root[len(home):].count(os.sep)
            if depth > max_depth:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in ("AppData", "node_modules", ".git", "site-packages", "Lib")]
            if ".obsidian" in dirs:
                found.append(root)
                dirs[:] = []
    except (PermissionError, OSError):
        pass
    return found

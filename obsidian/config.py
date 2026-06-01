"""ObsidianConfig dataclass + loader."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional


DEFAULT_VAULT_CANDIDATES = [
    r"C:\Users\User\Obsidian\TTRG",
    r"C:\Users\User\OneDrive\Documents\Obsidian Vault",
    r"C:\Users\User\Documents\ObsidianVault",
    r"C:\Users\User\Obsidian",
]


@dataclass
class ObsidianConfig:
    enabled: bool = True
    vault_path: str = r"C:\Users\User\Obsidian\TTRG"
    project_folder: str = "10-Projects/MEXC-Quant-Trading"
    write_trades: bool = True
    write_daily_summary: bool = True
    write_kpi: bool = True
    update_project_index: bool = True
    update_root_index: bool = False
    batch_interval_bars: int = 60
    auto_sync_on_trade: bool = True
    auto_sync_daily_hour: int = 0
    frontmatter_tags: list = field(default_factory=lambda: ["mexc", "auto-generated"])

    def is_valid(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "obsidian integration disabled"
        if not self.vault_path:
            return False, "vault_path is empty"
        if not os.path.isdir(self.vault_path):
            return False, f"vault_path does not exist: {self.vault_path}"
        project_full = os.path.join(self.vault_path, self.project_folder)
        if not os.path.isdir(project_full):
            return False, f"project folder does not exist: {project_full}"
        return True, "ok"

    def trades_dir(self) -> str:
        return os.path.join(self.vault_path, self.project_folder, "trades")

    def summaries_dir(self) -> str:
        return os.path.join(self.vault_path, self.project_folder, "daily-summaries")

    def project_dir(self) -> str:
        return os.path.join(self.vault_path, self.project_folder)

    def kpi_dir(self) -> str:
        return os.path.join(self.vault_path, "30-KPI")

    def to_dict(self) -> dict:
        return asdict(self)


_config_instance: Optional[ObsidianConfig] = None


def get_obsidian_config() -> ObsidianConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = ObsidianConfig()
    return _config_instance


def reset_obsidian_config() -> None:
    global _config_instance
    _config_instance = None

"""Supernet V3 training application layer."""

from typing import Any, Dict

from efnas.app.train_supernet_app import _load_yaml, _merge_overrides


def run_supernet_app_v3(config_path: str, overrides: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    """Run Supernet V3 training application flow."""
    base_config = _load_yaml(config_path)
    final_config = _merge_overrides(base_config, overrides)
    if dry_run:
        return {"exit_code": 0, "config": final_config}
    from efnas.engine.supernet_trainer_v3 import train_supernet

    exit_code = train_supernet(final_config)
    return {"exit_code": int(exit_code), "config": final_config}

"""Carga y fusion de configuracion (base + por-maquina).

Uso:
    from src.utils.config import load_config
    cfg = load_config()               # detecta la maquina por hostname
    print(cfg.machine.role, cfg.data_dir_abs)
"""
from __future__ import annotations

import socket
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS = _ROOT / "configs"


def _resolve_machine(machines_cfg: DictConfig, hostname: str) -> DictConfig:
    machines = machines_cfg.get("machines", {})
    if hostname in machines:
        m = machines[hostname]
    else:
        m = machines_cfg.get("default", {})
    return OmegaConf.create({**OmegaConf.to_container(m, resolve=True), "hostname": hostname})


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Devuelve la config fusionada: base.yaml + maquina detectada + overrides CLI."""
    base = OmegaConf.load(_CONFIGS / "base.yaml")
    machines_cfg = OmegaConf.load(_CONFIGS / "machines.yaml")

    hostname = socket.gethostname()
    machine = _resolve_machine(machines_cfg, hostname)

    cfg = OmegaConf.merge(base, {"machine": machine})

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    # Rutas absolutas utiles derivadas de la maquina.
    data_dir = Path(str(cfg.machine.data_dir))
    if not data_dir.is_absolute():
        data_dir = (_ROOT / data_dir).resolve()
    cfg.data_dir_abs = str(data_dir)

    ckpt_dir = Path(str(cfg.machine.checkpoints_dir))
    if not ckpt_dir.is_absolute():
        ckpt_dir = (_ROOT / ckpt_dir).resolve()
    cfg.checkpoints_dir_abs = str(ckpt_dir)

    return cfg


if __name__ == "__main__":
    print(OmegaConf.to_yaml(load_config()))

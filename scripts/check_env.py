"""Comprobacion de entorno: hostname, PyTorch, CUDA, VRAM y config resuelta.

Ejecutar en AMBAS maquinas tras instalar el entorno:
    python scripts/check_env.py
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    print(f"hostname            : {socket.gethostname()}")
    print(f"python              : {sys.version.split()[0]}")

    try:
        import torch
        print(f"torch               : {torch.__version__}")
        print(f"cuda available      : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda version        : {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"gpu[{i}]              : {p.name} ({p.total_memory / 1024**3:.1f} GB)")
    except ImportError:
        print("torch               : NO INSTALADO (ver README)")

    try:
        from src.utils.config import load_config
        cfg = load_config()
        print("-" * 50)
        print(f"machine role        : {cfg.machine.role}")
        print(f"data_dir            : {cfg.data_dir_abs}")
        print(f"batch_size          : {cfg.machine.batch_size}")
        print(f"smoke mode          : {cfg.machine.smoke}")
        if cfg.machine.role == "unknown":
            print("  [!] Hostname no reconocido: anadelo a configs/machines.yaml")
    except Exception as e:  # noqa: BLE001
        print(f"config              : ERROR ({e})")


if __name__ == "__main__":
    main()

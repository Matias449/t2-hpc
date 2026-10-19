#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ejecutar.py (Acceso directo raiz)
Lanza el orquestador universal ubicado en scripts/ejecutar_experimentos.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(BASE_DIR, "scripts", "ejecutar_experimentos.py")

if __name__ == "__main__":
    import subprocess
    cmd = [sys.executable, SCRIPT_PATH] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))

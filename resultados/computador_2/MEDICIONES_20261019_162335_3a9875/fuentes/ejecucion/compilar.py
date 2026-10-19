"""Compila cuatro binarios y registra comandos, versiones y hashes."""
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = next(p for p in ROOT.iterdir() if p.is_dir() and (p / "serial.cpp").exists())
BIN = HERE / "bin"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    BIN.mkdir(exist_ok=True)
    if platform.system() == "Darwin":
        cxx = os.environ.get("CXX", "/usr/bin/clang++")
        libomp = subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()
        arch = "-mcpu=native" if platform.machine() == "arm64" else "-march=native"
        flags = ["-O3", arch, "-std=c++17", "-Xpreprocessor", "-fopenmp",
                 "-I" + libomp + "/include", "-L" + libomp + "/lib",
                 "-Wl,-rpath," + libomp + "/lib", "-lomp"]
    else:
        cxx = os.environ.get("CXX", "g++")
        flags = ["-O3", "-march=native", "-std=c++17", "-fopenmp"]
    info = {"compiler": subprocess.check_output([cxx, "--version"], text=True),
            "commands": [], "sources": {}, "binaries": {}}
    for name, algo in [("serial", 0), ("datos", 1), ("tareas", 2), ("cuadrantes", 3)]:
        cmd = [cxx, *flags, "-DALGORITHM=" + str(algo), "-I" + str(SRC),
               str(HERE / "benchmark.cpp"), "-o", str(BIN / name)]
        print("Compilando", name, flush=True)
        subprocess.run(cmd, check=True)
        info["commands"].append(cmd)
        info["binaries"][name] = sha(BIN / name)
    for path in [HERE / "benchmark.cpp", *SRC.glob("*.cpp")]:
        info["sources"][str(path.relative_to(ROOT))] = sha(path)
    (HERE / "compilacion.json").write_text(json.dumps(info, indent=2))
    print("Compilación terminada")


if __name__ == "__main__":
    main()

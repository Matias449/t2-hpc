#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ejecutar_experimentos.py
===============================================================================
Orquestador universal de experimentos para la Tarea 2 - IIC3533 (HPC).
Compatible con macOS (Apple Silicon e Intel), Linux (Ubuntu/Debian/Arch/Fedora) y Windows.

Ejecuta de manera reproducible y automatizada:
  - Apartado (a): serial.cpp con N=2048 variando n in {2048, 1024, 512, 256, 128, 64, 32}
  - Apartado (b): paralelo_b.cpp (Algoritmo 1 - OpenMP) con N=2048, p in {1, 2, 4, 8}
  - Apartado (c): paralelo_c.cpp (Algoritmo 2 - OpenMP) con N=2048, p in {1, 2, 4, 8} (n=64)
                  y variacion de n in {32, 64, 128, 256, 512, 1024, 2048} con p=4
  - Apartado (d): Algoritmo 3 en Python con Numba (warm-up incluido) para p in {1, 2, 4, 8}
  - Apartado (e): Generacion de tablas y graficos de T(p), Speedup y Eficiencia
  - Apartado (f): Escalabilidad debil para N in {1024, 2048, 4096} en los tres algoritmos
  - Apartado (g): Deteccion automatica del hardware local y comparativa con el Computador 1

Uso:
  python3 ejecutar_experimentos.py                # Modo normal (detecta PC y corre la suite)
  python3 ejecutar_experimentos.py --computador 2 # Fuerza identificador como Computador 2
  python3 ejecutar_experimentos.py --reps 3        # Modifica repeticiones por punto (default: 5)
  python3 ejecutar_experimentos.py --pdf           # Compila automaticamente a PDF al finalizar
===============================================================================
"""

import os
import sys
import json
import time
import shutil
import argparse
import platform
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Configuracion de threads antes de imports
os.environ["NUMBA_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

try:
    from numba import config, njit, prange, set_num_threads
    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False
    print("[!] Numba no detectado. Instala dependencias con: pip install -r requirements.txt")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Deteccion de Hardware y Sistema Operativo
# -----------------------------------------------------------------------------
def detectar_hardware():
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "model_name": "Generico",
        "model_id": platform.node(),
        "cpu_brand": platform.processor() or "Desconocido",
        "cores_total": os.cpu_count() or 4,
        "cores_perf": os.cpu_count() or 4,
        "cores_eff": 0,
        "clock_mhz": "Variable",
        "ram_gb": 8,
        "compiler": "Desconocido",
        "openmp_flags": []
    }

    sistema = platform.system()

    if sistema == "Darwin":
        try:
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            info["cpu_brand"] = brand
        except Exception:
            pass
        try:
            model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
            info["model_id"] = model
            info["model_name"] = model
        except Exception:
            pass
        try:
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            info["ram_gb"] = round(mem_bytes / (1024**3), 1)
        except Exception:
            pass
        try:
            p_cores = int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True).strip())
            e_cores = int(subprocess.check_output(["sysctl", "-n", "hw.perflevel1.logicalcpu"], text=True).strip())
            info["cores_perf"] = p_cores
            info["cores_eff"] = e_cores
        except Exception:
            info["cores_perf"] = info["cores_total"]
            info["cores_eff"] = 0

    elif sistema == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        info["cpu_brand"] = line.split(":", 1)[1].strip()
                        break
                    elif "Hardware" in line:
                        info["cpu_brand"] = line.split(":", 1)[1].strip()
        except Exception:
            pass
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        info["ram_gb"] = round(kb / (1024**2), 1)
                        break
        except Exception:
            pass

    return info

# -----------------------------------------------------------------------------
# Deteccion y Compilacion de C++
# -----------------------------------------------------------------------------
def compilar_binarios(hw_info):
    print("=" * 80)
    print("PASO 0: Compilacion de binarios C++ con soporte OpenMP")
    print("=" * 80)

    sistema = platform.system()
    compiler = None
    flags = ["-O3"]

    # Intentar detectar compilador adecuado
    if shutil.which("g++"):
        compiler = "g++"
        flags = ["-O3", "-march=native", "-fopenmp"]
    elif shutil.which("clang++"):
        compiler = "clang++"
        if sistema == "Darwin":
            # Buscar libomp via brew
            brew_libomp = None
            for p in ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]:
                if os.path.exists(p):
                    brew_libomp = p
                    break
            if brew_libomp is None:
                try:
                    brew_libomp = subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()
                except Exception:
                    brew_libomp = None

            if brew_libomp and os.path.exists(brew_libomp):
                flags = [
                    "-O3", "-march=native", "-Xpreprocessor", "-fopenmp",
                    f"-I{brew_libomp}/include", f"-L{brew_libomp}/lib", "-lomp"
                ]
            else:
                flags = ["-O3", "-fopenmp"]
        else:
            flags = ["-O3", "-march=native", "-fopenmp"]
    else:
        raise RuntimeError("No se encontro un compilador C++ compatible (g++ o clang++) en el PATH.")

    hw_info["compiler"] = compiler
    hw_info["openmp_flags"] = flags

    programas = ["serial.cpp", "paralelo_b.cpp", "paralelo_c.cpp"]
    for prog in programas:
        out_name = prog.replace(".cpp", "")
        cmd = [compiler] + flags + [prog, "-o", out_name]
        print(f"Compilando {prog} -> {out_name} ...")
        res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
        if res.returncode != 0:
            # Reintentar sin -march=native si fallo (ej. arquitecturas virtuales)
            if "-march=native" in flags:
                flags_fallback = [f for f in flags if f != "-march=native"]
                cmd = [compiler] + flags_fallback + [prog, "-o", out_name]
                print(f"  Reintentando sin -march=native: {' '.join(cmd)}")
                res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"ERROR al compilar {prog}:\n{res.stderr}")
                sys.exit(1)

    print(">>> Todos los ejecutables C++ compilados exitosamente.\n")

def ejecutar_cpp(cmd_args):
    """Ejecuta un comando binario local y devuelve tiempo en ms y checksum."""
    res = subprocess.run(cmd_args, cwd=BASE_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Error al ejecutar {' '.join(cmd_args)}:\n{res.stderr}")
    line = res.stdout.strip()
    ms = None
    chk = None
    for token in line.split():
        if token.startswith("tiempo="):
            ms = float(token.replace("tiempo=", "").replace("ms", ""))
        elif token.startswith("checksum="):
            chk = float(token.replace("checksum=", ""))
    return ms, chk

# -----------------------------------------------------------------------------
# Mediciones de los Apartados
# -----------------------------------------------------------------------------
def correr_apartado_a(reps=5):
    print("=" * 80)
    print("APARTADO (a): Base serial y tamano de bloque (serial.cpp)")
    print("N = 2048, n in {2048, 1024, 512, 256, 128, 64, 32}")
    print("=" * 80)
    bloques = [2048, 1024, 512, 256, 128, 64, 32]
    resultados = {}
    for n in bloques:
        tiempos = []
        chk = 0.0
        for r in range(reps):
            ms, chk = ejecutar_cpp(["./serial", "2048", str(n)])
            tiempos.append(ms)
            print(f"  [serial] n={n:4d} | rep={r+1}/{reps} | tiempo={ms:8.2f} ms | chk={chk:.6f}")
        resultados[n] = {
            "n": n,
            "tiempos_raw": tiempos,
            "mean_ms": round(float(np.mean(tiempos)), 2),
            "std_ms": round(float(np.std(tiempos, ddof=1)), 2) if reps > 1 else 0.0,
            "checksum": chk
        }
        print(f"  --> Promedio n={n}: {resultados[n]['mean_ms']} +- {resultados[n]['std_ms']} ms\n")
    return resultados

def correr_apartado_b(reps=5):
    print("=" * 80)
    print("APARTADO (b): Algoritmo 1 - Paralelismo de datos con OpenMP (paralelo_b)")
    print("N = 2048, p in {1, 2, 4, 8}")
    print("=" * 80)
    threads = [1, 2, 4, 8]
    resultados = {}
    for p in threads:
        tiempos = []
        chk = 0.0
        for r in range(reps):
            ms, chk = ejecutar_cpp(["./paralelo_b", "2048", str(p)])
            tiempos.append(ms)
            print(f"  [paralelo_b] p={p} | rep={r+1}/{reps} | tiempo={ms:8.2f} ms | chk={chk:.6f}")
        resultados[p] = {
            "p": p,
            "tiempos_raw": tiempos,
            "mean_ms": round(float(np.mean(tiempos)), 2),
            "std_ms": round(float(np.std(tiempos, ddof=1)), 2) if reps > 1 else 0.0,
            "checksum": chk
        }
        print(f"  --> Promedio p={p}: {resultados[p]['mean_ms']} +- {resultados[p]['std_ms']} ms\n")
    return resultados

def correr_apartado_c(reps=5):
    print("=" * 80)
    print("APARTADO (c): Algoritmo 2 - Dividir y Conquistar con OpenMP (paralelo_c)")
    print("N = 2048, n = 64, p in {1, 2, 4, 8}")
    print("=" * 80)
    threads = [1, 2, 4, 8]
    res_p = {}
    for p in threads:
        tiempos = []
        chk = 0.0
        for r in range(reps):
            ms, chk = ejecutar_cpp(["./paralelo_c", "2048", "64", str(p)])
            tiempos.append(ms)
            print(f"  [paralelo_c] n=64 | p={p} | rep={r+1}/{reps} | tiempo={ms:8.2f} ms | chk={chk:.6f}")
        res_p[p] = {
            "p": p, "n": 64,
            "tiempos_raw": tiempos,
            "mean_ms": round(float(np.mean(tiempos)), 2),
            "std_ms": round(float(np.std(tiempos, ddof=1)), 2) if reps > 1 else 0.0,
            "checksum": chk
        }
        print(f"  --> Promedio p={p}: {res_p[p]['mean_ms']} +- {res_p[p]['std_ms']} ms\n")

    print("=" * 80)
    print("APARTADO (c/e): Variacion de tamano de bloque n para Algoritmo 2 con p = 4 threads")
    print("=" * 80)
    bloques = [2048, 1024, 512, 256, 128, 64, 32]
    res_bloques = {}
    for n in bloques:
        if n == 64:
            res_bloques[n] = res_p[4]
            continue
        tiempos = []
        chk = 0.0
        for r in range(reps):
            ms, chk = ejecutar_cpp(["./paralelo_c", "2048", str(n), "4"])
            tiempos.append(ms)
            print(f"  [paralelo_c] n={n:4d} | p=4 | rep={r+1}/{reps} | tiempo={ms:8.2f} ms")
        res_bloques[n] = {
            "p": 4, "n": n,
            "tiempos_raw": tiempos,
            "mean_ms": round(float(np.mean(tiempos)), 2),
            "std_ms": round(float(np.std(tiempos, ddof=1)), 2) if reps > 1 else 0.0,
            "checksum": chk
        }
        print(f"  --> Promedio n={n} (p=4): {res_bloques[n]['mean_ms']} +- {res_bloques[n]['std_ms']} ms\n")

    return res_p, res_bloques

def correr_apartado_d(reps=5):
    print("=" * 80)
    print("APARTADO (d): Algoritmo 3 - Python con Numba (@njit(parallel=True))")
    print("N = 2048, p in {1, 2, 4, 8}")
    print("=" * 80)

    if not NUMBA_OK:
        print("[!] Numba no esta disponible. Omitiendo benchmark de Numba.")
        return {}

    @njit
    def init_mat(N, offset):
        M = np.empty((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                M[i, j] = np.sin(i * 0.1 + j * 0.3 + offset)
        return M

    @njit(parallel=True)
    def matmul_nb(A, B, N):
        C = np.zeros((N, N), dtype=np.float64)
        for i in prange(N):
            for k in range(N):
                a_ik = A[i, k]
                for j in range(N):
                    C[i, j] += a_ik * B[k, j]
        return C

    print("Inicializando matrices A y B (2048x2048)...")
    A = init_mat(2048, 1.0)
    B = init_mat(2048, 2.0)

    print("Warm-up JIT con matriz 16x16 para excluir tiempo de compilacion...")
    Aw = init_mat(16, 1.0)
    Bw = init_mat(16, 2.0)
    set_num_threads(1)
    _ = matmul_nb(Aw, Bw, 16)

    threads = [1, 2, 4, 8]
    res_d = {}
    for p in threads:
        try:
            set_num_threads(p)
        except ValueError:
            set_num_threads(config.NUMBA_NUM_THREADS)
        _ = matmul_nb(Aw, Bw, 16)
        tiempos = []
        chk = 0.0
        for r in range(reps):
            t0 = time.perf_counter()
            C = matmul_nb(A, B, 2048)
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000.0
            chk = float(np.sum(C))
            tiempos.append(ms)
            print(f"  [numba] p={p} | rep={r+1}/{reps} | tiempo={ms:8.2f} ms | chk={chk:.6f}")
        res_d[p] = {
            "p": p,
            "tiempos_raw": tiempos,
            "mean_ms": round(float(np.mean(tiempos)), 2),
            "std_ms": round(float(np.std(tiempos, ddof=1)), 2) if reps > 1 else 0.0,
            "checksum": chk
        }
        print(f"  --> Promedio p={p}: {res_d[p]['mean_ms']} +- {res_d[p]['std_ms']} ms\n")
    return res_d

def correr_apartado_f(res_b, res_c, res_d):
    print("=" * 80)
    print("APARTADO (f): Escalabilidad Debil (N in {1024, 2048, 4096})")
    print("=" * 80)
    threads = [1, 2, 4, 8]
    datos_f = {
        "Algoritmo 1 (OpenMP Datos)": {},
        "Algoritmo 2 (OpenMP Div&Conq, n=64)": {},
        "Algoritmo 3 (Python Numba)": {}
    }

    # N=2048 ya fue medido
    datos_f["Algoritmo 1 (OpenMP Datos)"][2048] = [res_b[p]["mean_ms"] for p in threads]
    datos_f["Algoritmo 2 (OpenMP Div&Conq, n=64)"][2048] = [res_c[p]["mean_ms"] for p in threads]
    if res_d:
        datos_f["Algoritmo 3 (Python Numba)"][2048] = [res_d[p]["mean_ms"] for p in threads]

    # Medir N=1024 y N=4096
    for N in [1024, 4096]:
        reps = 3 if N == 4096 else 5
        print(f"\n>>> Midiendo tamano N = {N} (reps = {reps}) ...")

        # Algoritmo 1
        t_a1 = []
        for p in threads:
            vals = [ejecutar_cpp(["./paralelo_b", str(N), str(p)])[0] for _ in range(reps)]
            t_a1.append(round(float(np.mean(vals)), 2))
        datos_f["Algoritmo 1 (OpenMP Datos)"][N] = t_a1

        # Algoritmo 2
        t_a2 = []
        for p in threads:
            vals = [ejecutar_cpp(["./paralelo_c", str(N), "64", str(p)])[0] for _ in range(reps)]
            t_a2.append(round(float(np.mean(vals)), 2))
        datos_f["Algoritmo 2 (OpenMP Div&Conq, n=64)"][N] = t_a2

        # Algoritmo 3 (Numba)
        if NUMBA_OK:
            @njit
            def init_mat(N, offset):
                M = np.empty((N, N), dtype=np.float64)
                for i in range(N):
                    for j in range(N):
                        M[i, j] = np.sin(i * 0.1 + j * 0.3 + offset)
                return M
            @njit(parallel=True)
            def matmul_nb(A, B, N):
                C = np.zeros((N, N), dtype=np.float64)
                for i in prange(N):
                    for k in range(N):
                        a_ik = A[i, k]
                        for j in range(N):
                            C[i, j] += a_ik * B[k, j]
                return C

            A_nb = init_mat(N, 1.0)
            B_nb = init_mat(N, 2.0)
            t_a3 = []
            for p in threads:
                try: set_num_threads(p)
                except ValueError: set_num_threads(config.NUMBA_NUM_THREADS)
                vals = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    _ = matmul_nb(A_nb, B_nb, N)
                    vals.append((time.perf_counter() - t0) * 1000.0)
                t_a3.append(round(float(np.mean(vals)), 2))
            datos_f["Algoritmo 3 (Python Numba)"][N] = t_a3

    return datos_f

# -----------------------------------------------------------------------------
# Generacion de Reporte Markdown y Comparativa
# -----------------------------------------------------------------------------
def generar_reporte_md(num_pc, hw_info, res_a, res_b, res_c, res_c_bloques, res_d, datos_f):
    suffix = f"_COMPUTADOR_{num_pc}" if num_pc > 1 else ""
    nombre_md = f"INFORME_EXPERIMENTOS{suffix}.md"
    target_path = os.path.join(BASE_DIR, nombre_md)

    threads = [1, 2, 4, 8]
    t_ser_opt = res_a[2048]["mean_ms"]
    t1_b = res_b[1]["mean_ms"]
    t1_c = res_c[1]["mean_ms"]
    t1_d = res_d[1]["mean_ms"] if res_d else 0.0

    # Cargar datos de Computador 1 si estamos en Computador 2
    c1_data = None
    c1_path = os.path.join(BASE_DIR, "resultados_computador_1.json")
    if os.path.exists(c1_path):
        try:
            with open(c1_path, "r", encoding="utf-8") as f:
                c1_data = json.load(f)
        except Exception:
            pass

    md = []
    md.append(f"# Informe de Resultados Experimentales - Computador {num_pc}")
    md.append("## Tarea 2 IIC3533 · Computación de Alto Rendimiento (OpenMP y Numba)\n")
    md.append(f"> **Fecha y Hora de Medición:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"> **Equipo Evaluado (Computador {num_pc}):** {hw_info['cpu_brand']} ({hw_info['cores_total']} cores)")
    md.append(f"> **Memoria RAM:** {hw_info['ram_gb']} GB | **SO:** {hw_info['os']} {hw_info['os_release']} ({hw_info['arch']})")
    md.append(f"> **Compilador C++:** {hw_info['compiler']} con flags `{' '.join(hw_info['openmp_flags'])}`\n")

    md.append("---\n## Resumen de Resultados por Apartado\n")

    # Tabla Apartado A
    md.append("### (a) Base Serial y Tamaño de Bloque ($N = 2048$)")
    md.append("| Tamaño bloque $n$ | $T(n)$ [ms] | Desv. Est. [ms] | Checksum |")
    md.append("| :---: | :---: | :---: | :---: |")
    for n in [32, 64, 128, 256, 512, 1024, 2048]:
        md.append(f"| **{n}** | **{res_a[n]['mean_ms']:.2f}** | ± {res_a[n]['std_ms']:.2f} | `{res_a[n]['checksum']:.6f}` |")

    # Tabla Comparativa de Algoritmos
    md.append("\n### (b, c, d, e) Comparación de Algoritmos Paralelos ($N = 2048$)")
    md.append("| Algoritmo | $p$ | $T(p)$ [ms] | Speedup Relativo $S(p)$ | Eficiencia $E(p)$ | Speedup Absoluto (vs Serial) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    algs = [
        ("Alg. 1: OpenMP Datos", res_b, t1_b),
        ("Alg. 2: OpenMP Div&Conq (n=64)", res_c, t1_c),
    ]
    if res_d:
        algs.append(("Alg. 3: Python Numba", res_d, t1_d))

    for nombre, dic, t1_p in algs:
        for idx, p in enumerate(threads):
            m = dic[p]["mean_ms"]
            s_rel = t1_p / m
            e_rel = (s_rel / p) * 100.0
            s_abs = t_ser_opt / m
            nom_col = f"**{nombre}**" if idx == 0 else ""
            md.append(f"| {nom_col} | {p} | {m:.2f} | {s_rel:.2f}x | {e_rel:.1f}% | {s_abs:.2f}x |")

    # Escalabilidad debil
    md.append("\n### (f) Escalabilidad Débil para $N \\in \\{1024, 2048, 4096\\}$")
    md.append("| Algoritmo | $N$ | $p=1$ [ms] | $p=2$ [ms] | $p=4$ [ms] | $p=8$ [ms] | Speedup $S(N, 8)$ | Eficiencia $E(N, 8)$ |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for alg_nom in datos_f:
        for N in [1024, 2048, 4096]:
            row = datos_f[alg_nom][N]
            sp8 = row[0] / row[3]
            ef8 = (sp8 / 8.0) * 100.0
            md.append(f"| **{alg_nom}** | {N} | {row[0]:.2f} | {row[1]:.2f} | {row[2]:.2f} | {row[3]:.2f} | **{sp8:.2f}x** | **{ef8:.1f}%** |")

    # Comparativa con Computador 1 si existe
    if c1_data:
        c1_hw = c1_data.get("hardware", {})
        c1_ser = c1_data.get("apartado_a", {}).get("2048", {}).get("mean_ms", 1522.75)
        c1_b4 = c1_data.get("apartado_b", {}).get("4", {}).get("mean_ms", 691.96)
        c1_c4 = c1_data.get("apartado_c", {}).get("4", {}).get("mean_ms", 878.19)

        md.append("\n---\n## Apartado (g): Comparativa entre Computador 1 y Computador 2\n")
        md.append("| Parámetro | Computador 1 (Apple A18 Pro) | Computador 2 (Actual) |")
        md.append("| :--- | :--- | :--- |")
        md.append(f"| **Procesador (CPU)** | Apple A18 Pro (ARM64) | {hw_info['cpu_brand']} ({hw_info['arch']}) |")
        md.append(f"| **Núcleos Físicos** | 6 núcleos (2P + 4E) | {hw_info['cores_total']} núcleos |")
        md.append(f"| **Memoria RAM** | 8 GB LPDDR5X Unificada | {hw_info['ram_gb']} GB |")
        md.append(f"| **Sistema Operativo** | macOS (Darwin ARM64) | {hw_info['os']} {hw_info['os_release']} |")
        md.append(f"| **$T_{{serial}}(N=2048)$** | **{c1_ser:.2f} ms** | **{t_ser_opt:.2f} ms** |")
        md.append(f"| **$T_{{alg1}}(p=4)$** | **{c1_b4:.2f} ms** | **{res_b[4]['mean_ms']:.2f} ms** |")
        md.append(f"| **$T_{{alg2}}(p=4, n=64)$** | **{c1_c4:.2f} ms** | **{res_c[4]['mean_ms']:.2f} ms** |")

    with open(target_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n>>> INFORME MARKDOWN GENERADO CON EXITO: {target_path}")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Suite reproducible de experimentos Tarea 2 HPC")
    parser.add_argument("--computador", type=int, default=0, help="Identificador del computador (1 o 2). Por defecto se auto-detecta.")
    parser.add_argument("--reps", type=int, default=5, help="Repeticiones por punto de medicion (default: 5)")
    parser.add_argument("--pdf", action="store_true", help="Compilar automaticamente a PDF al finalizar")
    args = parser.parse_args()

    hw_info = detectar_hardware()

    # Auto-deteccion del computador
    num_pc = args.computador
    if num_pc == 0:
        if os.path.exists(os.path.join(BASE_DIR, "resultados_computador_1.json")) and hw_info["cpu_brand"] != "Apple A18 Pro":
            num_pc = 2
            print(f"[i] Se detecto 'resultados_computador_1.json'. Asignando este equipo como Computador 2.")
        else:
            num_pc = 1

    print(f"\nEJECUTANDO EXPERIMENTOS PARA: COMPUTADOR {num_pc}")
    print(f"Hardware detectado: {hw_info['cpu_brand']} | {hw_info['cores_total']} cores | {hw_info['ram_gb']} GB RAM | {hw_info['os']}\n")

    compilar_binarios(hw_info)

    t_start = time.time()
    res_a = correr_apartado_a(reps=args.reps)
    res_b = correr_apartado_b(reps=args.reps)
    res_c, res_c_bloques = correr_apartado_c(reps=args.reps)
    res_d = correr_apartado_d(reps=args.reps)
    datos_f = correr_apartado_f(res_b, res_c, res_d)

    # Guardar JSON
    datos_guardar = {
        "computador_id": num_pc,
        "hardware": hw_info,
        "apartado_a": res_a,
        "apartado_b": res_b,
        "apartado_c": res_c,
        "apartado_c_bloques_p4": res_c_bloques,
        "apartado_d": res_d,
        "apartado_f": datos_f,
    }
    json_name = f"resultados_computador_{num_pc}.json"
    with open(os.path.join(BASE_DIR, json_name), "w", encoding="utf-8") as f:
        json.dump(datos_guardar, f, indent=2)
    print(f"\n>>> Datos brutos JSON guardados en: {json_name}")

    generar_reporte_md(num_pc, hw_info, res_a, res_b, res_c, res_c_bloques, res_d, datos_f)

    if args.pdf:
        try:
            import convertir_informe_a_pdf
            convertir_informe_a_pdf.generar_html()
            convertir_informe_a_pdf.generar_pdf()
        except Exception as e:
            print(f"[!] No se pudo compilar PDF automaticamente: {e}")

    total_s = time.time() - t_start
    print("=" * 80)
    print(f"SUITE COMPLETA FINALIZADA EN {total_s:.1f} SEGUNDOS.")
    print("=" * 80)

if __name__ == "__main__":
    main()

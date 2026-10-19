#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generar_graficos_comparativos.py
Genera figuras comparativas de alta resolucion (300 DPI) contrastando los resultados
del Computador 1 (Apple A18 Pro) y del Computador 2 (Apple M5).
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
C1_JSON = os.path.join(REPO_ROOT, "resultados", "computador_1", "resultados_computador_1.json")
C2_CSV = os.path.join(REPO_ROOT, "resultados", "computador_2", "MEDICIONES_20261019_162335_3a9875", "resumen.csv")
OUT_DIR = os.path.join(REPO_ROOT, "resultados", "comparativas")
os.makedirs(OUT_DIR, exist_ok=True)

def cargar_c1():
    with open(C1_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def cargar_c2():
    c2 = {}
    with open(C2_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alg = row["algorithm"]
            N = int(row["N"])
            n = int(row["n"])
            p = int(row["p"])
            mean_ms = float(row["mean_ms"])
            std_ms = float(row["std_ms"])
            sp = float(row["speedup"])
            ef = float(row["efficiency"]) * 100.0

            key = (alg, N, n, p)
            c2[key] = {
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "speedup": sp,
                "efficiency": ef
            }
    return c2

def generar_graficos():
    c1 = cargar_c1()
    c2 = cargar_c2()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
    })

    color_c1 = "#1f77b4"  # Azul (A18 Pro)
    color_c2 = "#d95f02"  # Naranja (M5)
    threads = [1, 2, 4, 8]
    p_arr = np.array(threads, dtype=float)
    bloques = [32, 64, 128, 256, 512, 1024, 2048]

    # =========================================================================
    # 1. FIGURA COMPARATIVA 1: Apartado (a) - Serial vs Bloque n
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    t_a_c1 = [c1["apartado_a"][str(n)]["mean_ms"] for n in bloques]
    s_a_c1 = [c1["apartado_a"][str(n)]["std_ms"] for n in bloques]

    t_a_c2 = [c2[("serial", 2048, n, 1)]["mean_ms"] for n in bloques]
    s_a_c2 = [c2[("serial", 2048, n, 1)]["std_ms"] for n in bloques]

    ax.errorbar([str(n) for n in bloques], t_a_c1, yerr=s_a_c1, fmt="-o", color=color_c1,
                linewidth=2, markersize=7, capsize=4, label="Computador 1: Apple A18 Pro (2P + 4E)")
    ax.errorbar([str(n) for n in bloques], t_a_c2, yerr=s_a_c2, fmt="-s", color=color_c2,
                linewidth=2, markersize=7, capsize=4, label="Computador 2: Apple M5 (4P + 6E)")

    ax.set_title("Apartado (a): Tiempo Serial vs. Tamaño de Bloque $n$ ($N = 2048$)")
    ax.set_xlabel("Tamaño de bloque de hoja $n$")
    ax.set_ylabel("Tiempo de ejecución (ms)")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "comparativa_1_apartado_a_bloques.png")
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    print(f"[✓] Guardado: {p1}")

    # =========================================================================
    # 2. FIGURA COMPARATIVA 2: Tiempos T(p) por Algoritmo (3 paneles)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)

    # Algoritmo 1: OpenMP Datos
    ax = axes[0]
    t_b_c1 = [c1["apartado_b"][str(p)]["mean_ms"] for p in threads]
    t_b_c2 = [c2[("datos", 2048, 2048, p)]["mean_ms"] for p in threads]
    ax.plot(p_arr, t_b_c1, "-o", color=color_c1, linewidth=2, label="Apple A18 Pro")
    ax.plot(p_arr, t_b_c2, "-s", color=color_c2, linewidth=2, label="Apple M5")
    ax.set_title("Alg. 1: OpenMP Datos ($N=2048$)")
    ax.set_xlabel("Threads ($p$)")
    ax.set_ylabel("Tiempo de ejecución (ms)")
    ax.set_xticks(threads)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()

    # Algoritmo 2: Dividir y Conquistar (n=64)
    ax = axes[1]
    t_c_c1 = [c1["apartado_c"][str(p)]["mean_ms"] for p in threads]
    t_c_c2 = [c2[("cuadrantes", 2048, 64, p)]["mean_ms"] for p in threads]
    ax.plot(p_arr, t_c_c1, "-o", color=color_c1, linewidth=2, label="Apple A18 Pro")
    ax.plot(p_arr, t_c_c2, "-s", color=color_c2, linewidth=2, label="Apple M5")
    ax.set_title("Alg. 2: OpenMP Div&Conq ($n=64$)")
    ax.set_xlabel("Threads ($p$)")
    ax.set_xticks(threads)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()

    # Algoritmo 3: Python Numba
    ax = axes[2]
    t_d_c1 = [c1["apartado_d"][str(p)]["mean_ms"] for p in threads]
    t_d_c2 = [c2[("numba", 2048, 2048, p)]["mean_ms"] for p in threads]
    ax.plot(p_arr, t_d_c1, "-o", color=color_c1, linewidth=2, label="Apple A18 Pro")
    ax.plot(p_arr, t_d_c2, "-s", color=color_c2, linewidth=2, label="Apple M5")
    ax.set_title("Alg. 3: Python Numba ($N=2048$)")
    ax.set_xlabel("Threads ($p$)")
    ax.set_xticks(threads)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()

    fig.suptitle("Comparativa de Tiempos de Ejecución $T(p)$: Computador 1 vs. Computador 2", y=1.02)
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "comparativa_2_tiempos_ejecucion.png")
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    print(f"[✓] Guardado: {p2}")

    # =========================================================================
    # 3. FIGURA COMPARATIVA 3: Speedup Relativo S(p) y Eficiencia E(p)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    # Speedup
    ax = axes[0]
    ax.plot(p_arr, p_arr, "--", color="#555", linewidth=1.5, label="Ideal ($S=p$)")
    # Alg 1
    ax.plot(p_arr, t_b_c1[0] / np.array(t_b_c1), "-o", color=color_c1, label="Alg. 1 (A18 Pro)")
    ax.plot(p_arr, t_b_c2[0] / np.array(t_b_c2), "--o", color=color_c2, label="Alg. 1 (M5)")
    # Alg 2
    ax.plot(p_arr, t_c_c1[0] / np.array(t_c_c1), "-s", color="#1b9e77", label="Alg. 2 (A18 Pro)")
    ax.plot(p_arr, t_c_c2[0] / np.array(t_c_c2), "--s", color="#7570b3", label="Alg. 2 (M5)")
    ax.set_title("Speedup Relativo $S(p) = T(1)/T(p)$ ($N=2048$)")
    ax.set_xlabel("Threads ($p$)")
    ax.set_ylabel("Speedup $S(p)$")
    ax.set_xticks(threads)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(fontsize=8.5)

    # Eficiencia
    ax = axes[1]
    ax.axhline(100.0, linestyle="--", color="#555", linewidth=1.5, label="Ideal (100%)")
    ax.plot(p_arr, (t_b_c1[0] / np.array(t_b_c1) / p_arr) * 100.0, "-o", color=color_c1, label="Alg. 1 (A18 Pro)")
    ax.plot(p_arr, (t_b_c2[0] / np.array(t_b_c2) / p_arr) * 100.0, "--o", color=color_c2, label="Alg. 1 (M5)")
    ax.plot(p_arr, (t_c_c1[0] / np.array(t_c_c1) / p_arr) * 100.0, "-s", color="#1b9e77", label="Alg. 2 (A18 Pro)")
    ax.plot(p_arr, (t_c_c2[0] / np.array(t_c_c2) / p_arr) * 100.0, "--s", color="#7570b3", label="Alg. 2 (M5)")
    ax.set_title("Eficiencia Relativa $E(p)$ (%) ($N=2048$)")
    ax.set_xlabel("Threads ($p$)")
    ax.set_ylabel("Eficiencia (%)")
    ax.set_xticks(threads)
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(fontsize=8.5)

    fig.tight_layout()
    p3 = os.path.join(OUT_DIR, "comparativa_3_speedup_eficiencia.png")
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    print(f"[✓] Guardado: {p3}")

    # =========================================================================
    # 4. FIGURA COMPARATIVA 4: Barrido de Bloques en Algoritmo 2 (p=4)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    t_c_bloques_c1 = [c1["apartado_c_bloques_p4"][str(n)]["mean_ms"] for n in bloques]
    t_c_bloques_c2 = [c2[("cuadrantes", 2048, n, 4)]["mean_ms"] for n in bloques]

    ax.plot([str(n) for n in bloques], t_c_bloques_c1, "-o", color=color_c1, linewidth=2, markersize=7,
            label="Computador 1: Apple A18 Pro (Óptimo en n=1024 / n=256)")
    ax.plot([str(n) for n in bloques], t_c_bloques_c2, "-s", color=color_c2, linewidth=2, markersize=7,
            label="Computador 2: Apple M5 (Óptimo en n=512 / n=1024)")

    ax.set_title("Apartado (e): Impacto del Tamaño de Bloque $n$ en Algoritmo 2 ($p=4$, $N=2048$)")
    ax.set_xlabel("Tamaño de bloque de hoja $n$")
    ax.set_ylabel("Tiempo de ejecución (ms)")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    p4 = os.path.join(OUT_DIR, "comparativa_4_alg2_bloques_p4.png")
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    print(f"[✓] Guardado: {p4}")

    # =========================================================================
    # 5. FIGURA COMPARATIVA 5: Eficiencia Débil Iso-granular
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    labels = ["Alg. 1 (OpenMP Datos)", "Alg. 2 (OpenMP Div&Conq)", "Alg. 3 (Python Numba)"]
    x = np.arange(len(labels))
    width = 0.35

    # Par (1024, 1) -> (2048, 8)
    ef_iso_c1 = [
        (c1["apartado_f"]["Algoritmo 1 (OpenMP Datos)"]["1024"][0] / c1["apartado_f"]["Algoritmo 1 (OpenMP Datos)"]["2048"][3]) * 100.0,
        (c1["apartado_f"]["Algoritmo 2 (OpenMP Div&Conq, n=64)"]["1024"][0] / c1["apartado_f"]["Algoritmo 2 (OpenMP Div&Conq, n=64)"]["2048"][3]) * 100.0,
        (c1["apartado_f"]["Algoritmo 3 (Python Numba)"]["1024"][0] / c1["apartado_f"]["Algoritmo 3 (Python Numba)"]["2048"][3]) * 100.0,
    ]

    ef_iso_c2 = [
        (c2[("datos", 1024, 1024, 1)]["mean_ms"] / c2[("datos", 2048, 2048, 8)]["mean_ms"]) * 100.0,
        (c2[("cuadrantes", 1024, 64, 1)]["mean_ms"] / c2[("cuadrantes", 2048, 64, 8)]["mean_ms"]) * 100.0,
        (c2[("numba", 1024, 1024, 1)]["mean_ms"] / c2[("numba", 2048, 2048, 8)]["mean_ms"]) * 100.0,
    ]

    rects1 = ax.bar(x - width/2, ef_iso_c1, width, label="Computador 1 (Apple A18 Pro, 6 cores)", color=color_c1)
    rects2 = ax.bar(x + width/2, ef_iso_c2, width, label="Computador 2 (Apple M5, 10 cores)", color=color_c2)

    ax.axhline(100.0, linestyle="--", color="#555", linewidth=1.2, label="Ideal (100%)")
    ax.set_ylabel("Eficiencia Débil Iso-granular (%)")
    ax.set_title("Apartado (f): Eficiencia Débil $[(N=1024, p=1) \\to (N=2048, p=8)]$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    ax.legend()

    # Anotar valores en las barras
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    p5 = os.path.join(OUT_DIR, "comparativa_5_escalabilidad_debil.png")
    fig.savefig(p5, dpi=300)
    plt.close(fig)
    print(f"[✓] Guardado: {p5}")

if __name__ == "__main__":
    generar_graficos()

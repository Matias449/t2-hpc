#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tarea 2 IIC3533 - Apartado (e): Comparacion y Escalamiento
==========================================================
Genera los graficos y tablas comparativas de:
  1. Tiempo de ejecucion T(p) en funcion de p (con barras de desviacion estandar).
  2. Speedup S(p) = T(1) / T(p) comparado con la curva ideal S_ideal(p) = p.
  3. Eficiencia E(p) = S(p) / p comparada con la curva ideal E_ideal(p) = 100%.
  4. Variacion del tiempo de ejecucion del Algoritmo 2 al cambiar el tamano de bloque n,
     comparado con la version serial (Apartado a).

INSTRUCCIONES:
Solo edita los valores en la seccion "1. DATOS MANUALES" de abajo y ejecuta:
    python3 graficos_apartado_e.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DATOS MANUALES (Completa o actualiza solo los valores marcados con <-- )
# ==============================================================================

THREADS = [1, 2, 4, 8]

# Mejor tiempo serial de serial.cpp (Tabla 1 del informe: n = 2048 -> 1372.96 ms)
T_SERIAL_OPTIMO = 1372.96

# --- Algoritmo 1: OpenMP Paralelismo de Datos (Ya cargado desde Tabla 2 del docx) ---
T_ALG1_OPENMP   = [1481.72, 1103.12, 583.88, 489.36]
STD_ALG1_OPENMP = [5.98,    24.23,   23.19,  15.14]

# --- Algoritmo 2: OpenMP Dividir y Conquistar, n = 64 (Ya cargado desde Tabla 3 del docx) ---
T_ALG2_DIVCONQ   = [1696.46, 900.14, 508.46, 510.31]
STD_ALG2_DIVCONQ = [3.32,    4.56,   12.86,  2.93]

# --- Algoritmo 3: Python con Numba (Mediciones reales N=2048, 5 reps) ---
T_ALG3_NUMBA   = [1614.07, 1122.88, 772.41, 672.34]
STD_ALG3_NUMBA = [6.87,    14.59,   19.27,  37.62]

# --- Eleccion de T(1) para calcular Speedup S(p) = T(1)/T(p) y Eficiencia E(p) = S(p)/p ---
# Opciones:
#   "propio"        -> Usa el T(1) de cada algoritmo con 1 thread (consistente con Tablas 2 y 3 del docx).
#   "serial_optimo" -> Usa T_SERIAL_OPTIMO (1372.96 ms, serial.cpp con n=2048) para los tres algoritmos.
CRITERIO_T1 = "propio"

# --- Variacion del tamano de bloque n (para el grafico de bloques del Algoritmo 2) ---
BLOQUES_N = [32, 64, 128, 256, 512, 1024, 2048]

# Tiempos seriales por bloque n (Ya cargados desde Tabla 1 del docx, ordenados de n=32 a n=2048)
T_SERIAL_POR_BLOQUE = [2150.98, 1678.20, 1726.17, 1693.92, 1798.47, 2388.85, 1372.96]

# Tiempos del Algoritmo 2 con p = 4 threads variando n (mediciones reales)
T_ALG2_POR_BLOQUE_P4 = [
    1123.09,  # n = 32
    878.19,   # n = 64
    794.39,   # n = 128
    785.71,   # n = 256
    826.94,   # n = 512
    761.77,   # n = 1024
    1526.86,  # n = 2048
]


# ==============================================================================
# 2. CALCULO DE METRICAS Y GENERACION DE TABLAS EN CONSOLA
# ==============================================================================

def calcular_metricas(tiempos, threads, modo_t1="propio", t_serial_opt=1372.96):
    t_arr = np.array(tiempos, dtype=float)
    p_arr = np.array(threads, dtype=float)
    t1 = t_arr[0] if modo_t1 == "propio" else float(t_serial_opt)
    speedup = t1 / t_arr
    eficiencia = (speedup / p_arr) * 100.0
    return t1, speedup, eficiencia


def imprimir_resumen():
    algoritmos = [
        ("Algoritmo 1 (OpenMP Datos)", T_ALG1_OPENMP, STD_ALG1_OPENMP),
        ("Algoritmo 2 (OpenMP Div&Conq, n=64)", T_ALG2_DIVCONQ, STD_ALG2_DIVCONQ),
        ("Algoritmo 3 (Python Numba)", T_ALG3_NUMBA, STD_ALG3_NUMBA),
    ]

    for modo in ["propio", "serial_optimo"]:
        titulo_modo = (
            "T(1) PROPIO DE CADA ALGORITMO (Speedup Relativo)"
            if modo == "propio"
            else f"T(1) SERIAL OPTIMO FIJO = {T_SERIAL_OPTIMO:.2f} ms (Speedup Absoluto)"
        )
        print("\n" + "=" * 95)
        print(f"TABLA COMPARATIVA APARTADO (e) - CRITERIO: {titulo_modo}")
        print("=" * 95)
        print(
            f"{'Algoritmo':<36} | {'p':^3} | {'T(p) [ms]':^12} | {'Desv [ms]':^10} | "
            f"{'Speedup S(p)':^12} | {'Eficiencia E(p)':^15}"
        )
        print("-" * 95)
        for nombre, t_list, std_list in algoritmos:
            _, s_arr, e_arr = calcular_metricas(t_list, THREADS, modo, T_SERIAL_OPTIMO)
            for idx, p in enumerate(THREADS):
                alg_col = nombre if idx == 0 else ""
                print(
                    f"{alg_col:<36} | {p:^3d} | {t_list[idx]:^12.2f} | {std_list[idx]:^10.2f} | "
                    f"{s_arr[idx]:^12.2f} | {e_arr[idx]:^14.1f}%"
                )
            print("-" * 95)


# ==============================================================================
# 3. GENERACION DE GRAFICOS (ALTA RESOLUCION PARA EL INFORME)
# ==============================================================================

def generar_graficos():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "figure.titlesize": 14,
    })

    p_arr = np.array(THREADS, dtype=float)
    _, s_alg1, e_alg1 = calcular_metricas(T_ALG1_OPENMP, THREADS, CRITERIO_T1, T_SERIAL_OPTIMO)
    _, s_alg2, e_alg2 = calcular_metricas(T_ALG2_DIVCONQ, THREADS, CRITERIO_T1, T_SERIAL_OPTIMO)
    _, s_alg3, e_alg3 = calcular_metricas(T_ALG3_NUMBA, THREADS, CRITERIO_T1, T_SERIAL_OPTIMO)

    # Colores y estilos distinguibles
    c1, c2, c3, c_ideal = "#1f77b4", "#d95f02", "#2ca02c", "#333333"

    # --------------------------------------------------------------------------
    # FIGURA 1: Panel triple con T(p), S(p) y E(p) en funcion de p
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)

    # (1a) Tiempo de ejecucion T(p)
    ax = axes[0]
    t_ideal_ref = T_SERIAL_OPTIMO / p_arr
    ax.plot(p_arr, t_ideal_ref, "--", color=c_ideal, linewidth=1.8, label="Curva ideal ($T_{serial}/p$)")
    ax.errorbar(p_arr, T_ALG1_OPENMP, yerr=STD_ALG1_OPENMP, fmt="-o", color=c1, linewidth=2, capsize=4, label="Alg. 1: OpenMP Datos")
    ax.errorbar(p_arr, T_ALG2_DIVCONQ, yerr=STD_ALG2_DIVCONQ, fmt="-s", color=c2, linewidth=2, capsize=4, label="Alg. 2: OpenMP Div&Conq ($n=64$)")
    ax.errorbar(p_arr, T_ALG3_NUMBA, yerr=STD_ALG3_NUMBA, fmt="-^", color=c3, linewidth=2, capsize=4, label="Alg. 3: Python Numba")
    ax.set_title("Tiempo de Ejecución $T(p)$ vs. Threads")
    ax.set_xlabel("Número de threads ($p$)")
    ax.set_ylabel("Tiempo promedio (ms)")
    ax.set_xticks(THREADS)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper right")

    # (1b) Speedup S(p)
    ax = axes[1]
    ax.plot(p_arr, p_arr, "--", color=c_ideal, linewidth=1.8, label="Speedup ideal ($S(p) = p$)")
    ax.plot(p_arr, s_alg1, "-o", color=c1, linewidth=2, markersize=6, label="Alg. 1: OpenMP Datos")
    ax.plot(p_arr, s_alg2, "-s", color=c2, linewidth=2, markersize=6, label="Alg. 2: OpenMP Div&Conq ($n=64$)")
    ax.plot(p_arr, s_alg3, "-^", color=c3, linewidth=2, markersize=6, label="Alg. 3: Python Numba")
    ax.set_title("Speedup $S(p)$ vs. Threads")
    ax.set_xlabel("Número de threads ($p$)")
    ax.set_ylabel("Speedup $S(p) = T(1) / T(p)$")
    ax.set_xticks(THREADS)
    ax.set_ylim(0, max(THREADS) + 0.5)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper left")

    # (1c) Eficiencia E(p)
    ax = axes[2]
    ax.axhline(100.0, linestyle="--", color=c_ideal, linewidth=1.8, label="Eficiencia ideal (100 %)")
    ax.plot(p_arr, e_alg1, "-o", color=c1, linewidth=2, markersize=6, label="Alg. 1: OpenMP Datos")
    ax.plot(p_arr, e_alg2, "-s", color=c2, linewidth=2, markersize=6, label="Alg. 2: OpenMP Div&Conq ($n=64$)")
    ax.plot(p_arr, e_alg3, "-^", color=c3, linewidth=2, markersize=6, label="Alg. 3: Python Numba")
    ax.set_title("Eficiencia Paralela $E(p)$ vs. Threads")
    ax.set_xlabel("Número de threads ($p$)")
    ax.set_ylabel("Eficiencia $E(p) = S(p)/p$ (%)")
    ax.set_xticks(THREADS)
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="lower left")

    fig.suptitle(f"Comparación y Escalamiento para $N = 2048$ (Criterio $T(1)$: {CRITERIO_T1})", fontweight="bold", y=1.02)
    plt.tight_layout()
    out_panel = "apartado_e_escalamiento_panel.png"
    fig.savefig(out_panel, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Grafico guardado: {out_panel}")

    # --------------------------------------------------------------------------
    # FIGURA 2: Efecto del tamano de bloque n en Algoritmo 2 vs. Base Serial
    # --------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    x_idx = np.arange(len(BLOQUES_N))
    x_labels = [str(n) for n in BLOQUES_N]

    ax2.plot(x_idx, T_SERIAL_POR_BLOQUE, "-o", color="#555555", linewidth=2, markersize=6, label="Base Serial (1 thread)")
    ax2.plot(x_idx, T_ALG2_POR_BLOQUE_P4, "-s", color=c2, linewidth=2.2, markersize=7, label="Algoritmo 2: Dividir y Conquistar ($p = 4$)")

    # Destacar el minimo del Algoritmo 2
    idx_min = int(np.argmin(T_ALG2_POR_BLOQUE_P4))
    n_opt = BLOQUES_N[idx_min]
    t_opt = T_ALG2_POR_BLOQUE_P4[idx_min]
    ax2.annotate(
        f"Óptimo $n={n_opt}$\n({t_opt:.2f} ms)",
        xy=(idx_min, t_opt),
        xytext=(idx_min + 0.3, t_opt + 220),
        arrowprops=dict(arrowstyle="->", color=c2, lw=1.5),
        fontsize=10,
        fontweight="bold",
        color=c2,
    )

    ax2.set_title("Impacto del Tamaño de Bloque ($n$) en el Tiempo de Ejecución ($N = 2048$)")
    ax2.set_xlabel("Tamaño de bloque hoja $n$ (escala $\\log_2$)")
    ax2.set_ylabel("Tiempo promedio de ejecución (ms)")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(x_labels)
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    out_bloques = "apartado_e_bloques_alg2.png"
    fig2.savefig(out_bloques, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[OK] Grafico guardado: {out_bloques}")


if __name__ == "__main__":
    imprimir_resumen()
    generar_graficos()
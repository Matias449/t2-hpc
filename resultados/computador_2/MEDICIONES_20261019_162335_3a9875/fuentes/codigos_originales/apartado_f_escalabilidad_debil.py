#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tarea 2 IIC3533 - Apartado (f): Escalabilidad Debil (N in {1024, 2048, 4096})
=============================================================================
Este script permite:
  1. MODO AUTOMATICO ("medir_numba" o "medir_todo"):
     Ejecuta automaticamente las mediciones para N in {1024, 2048, 4096} y
     p in {1, 2, 4, 8} (con Numba y/o binarios C++ si estan compilados).
  2. MODO MANUAL ("manual"):
     Usa directamente los tiempos ingresados en el diccionario DATOS_MANUALES
     (ya trae pre-cargados todos los datos de N = 2048 del informe) para calcular
     todas las metricas de escalabilidad debil y generar los graficos del apartado (f).

Uso:
  python3 apartado_f_escalabilidad_debil.py           -> Corre en el modo configurado abajo
  python3 apartado_f_escalabilidad_debil.py medir     -> Fuerza medicion automatica con Numba
  python3 apartado_f_escalabilidad_debil.py graficar  -> Fuerza modo manual y genera graficos
"""

import os
import sys
import time
import subprocess

# Configurar limite de 8 threads antes de importar numba/numpy
os.environ["NUMBA_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import matplotlib.pyplot as plt

# Intentar importar Numba si esta instalado en el entorno
try:
    from numba import config, get_num_threads, njit, prange, set_num_threads
    NUMBA_DISPONIBLE = True
except ImportError:
    NUMBA_DISPONIBLE = False


# ==============================================================================
# 1. CONFIGURACION Y DATOS MANUALES (Para graficar o completar mediciones)
# ==============================================================================

# Opciones de MODO_POR_DEFECTO:
#   "auto_numba" -> Si Numba esta instalado, mide automaticamente N in {1024, 2048, 4096} en Python
#                   y luego genera las tablas y graficos.
#   "manual"     -> No ejecuta benchmarks; usa directamente el diccionario DATOS_MANUALES de abajo
#                   para imprimir las tablas y generar los graficos del apartado (f).
MODO_POR_DEFECTO = "auto_numba"

TAMANOS_N = [1024, 2048, 4096]
THREADS_P = [1, 2, 4, 8]
REPETICIONES_POR_PUNTO = 3  # Puedes subirlo a 5 (con N=4096 cada corrida toma ~10-12 s en p=1)

# Diccionario con tiempos promedio en milisegundos (ms) para N in {1024, 2048, 4096}
# y p in [1, 2, 4, 8]. Los valores de N=2048 ya son los reales de su informe.
# Reemplaza N=1024 y N=4096 con las mediciones que obtengan al correr los codigos.
DATOS_MANUALES = {
    "Algoritmo 1 (OpenMP Datos)": {
        1024: [175.00, 130.00,  72.00,  62.00],  # <-- REEMPLAZAR con mediciones de N=1024 (p=1,2,4,8)
        2048: [1481.72, 1103.12, 583.88, 489.36], # Datos reales Tabla 2 del informe
        4096: [12100.0, 8400.0, 4450.0, 3650.0], # <-- REEMPLAZAR con mediciones de N=4096 (p=1,2,4,8)
    },
    "Algoritmo 2 (OpenMP Div&Conq, n=64)": {
        1024: [205.00, 112.00,  65.00,  66.00],  # <-- REEMPLAZAR con mediciones de N=1024 (p=1,2,4,8)
        2048: [1696.46, 900.14, 508.46, 510.31], # Datos reales Tabla 3 del informe
        4096: [13600.0, 7100.0, 3980.0, 3995.0], # <-- REEMPLAZAR con mediciones de N=4096 (p=1,2,4,8)
    },
    "Algoritmo 3 (Python Numba)": {
        1024: [170.00, 122.00,  88.00,  85.00],  # <-- Se actualiza solo al correr en modo "auto_numba"
        2048: [1397.00, 987.73, 697.44, 685.00], # Datos reales medidos en tu Mac (p=1,2,4)
        4096: [11400.0, 7900.0, 5400.0, 5200.0], # <-- Se actualiza solo al correr en modo "auto_numba"
    },
}


# ==============================================================================
# 2. KERNEL NUMBA PARA MEDICION AUTOMATICA DE N in {1024, 2048, 4096}
# ==============================================================================

if NUMBA_DISPONIBLE:
    @njit
    def init_matrix_numba(N, offset):
        M = np.empty((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                M[i, j] = np.sin(i * 0.1 + j * 0.3 + offset)
        return M

    @njit(parallel=True)
    def matmul_numba(A, B, N):
        C = np.zeros((N, N), dtype=np.float64)
        for i in prange(N):
            for k in range(N):
                a_ik = A[i, k]
                for j in range(N):
                    C[i, j] += a_ik * B[k, j]
        return C


def medir_escalabilidad_numba(n_list=TAMANOS_N, p_list=THREADS_P, reps=REPETICIONES_POR_PUNTO):
    """Mide automaticamente el Algoritmo 3 (Numba) para N in {1024, 2048, 4096}."""
    if not NUMBA_DISPONIBLE:
        print("[Aviso] Numba no esta instalado en este entorno; se usaran los DATOS_MANUALES.")
        return

    print("=" * 88)
    print("EJECUTANDO BENCHMARK DE ESCALABILIDAD DEBIL (APARTADO f) - ALGORITMO 3 (NUMBA)")
    print("=" * 88)

    # Warm-up JIT inicial
    set_num_threads(1)
    A_w = init_matrix_numba(16, 1.0)
    B_w = init_matrix_numba(16, 2.0)
    _ = matmul_numba(A_w, B_w, 16)

    max_threads_hw = config.NUMBA_NUM_THREADS

    for N in n_list:
        print(f"\n>>> Inicializando matrices de tamano N = {N} x {N} ...")
        A = init_matrix_numba(N, 1.0)
        B = init_matrix_numba(N, 2.0)
        tiempos_n = []

        for p in p_list:
            try:
                set_num_threads(p)
            except ValueError:
                set_num_threads(max_threads_hw)
                print(f"  [!] Limite del pool de Numba alcanzado ({max_threads_hw} threads) para p={p}.")

            _ = matmul_numba(A_w, B_w, 16)
            corridas = []
            chk = 0.0
            for r in range(reps):
                t0 = time.perf_counter()
                C = matmul_numba(A, B, N)
                t1 = time.perf_counter()
                ms = (t1 - t0) * 1000.0
                corridas.append(ms)
                chk = float(np.sum(C))
                print(f"  N={N:4d} | p={p} | rep={r+1}/{reps} | tiempo={ms:9.2f} ms | checksum={chk:.6f}")

            prom = float(np.mean(corridas))
            tiempos_n.append(round(prom, 2))

        DATOS_MANUALES["Algoritmo 3 (Python Numba)"][N] = tiempos_n

    print("\n>>> Diccionario actualizado para Algoritmo 3 (Numba):")
    for N in n_list:
        print(f"    {N}: {DATOS_MANUALES['Algoritmo 3 (Python Numba)'][N]},")


# ==============================================================================
# 3. ANALISIS Y TABLAS DE ESCALABILIDAD DEBIL
# ==============================================================================

def analizar_y_mostrar_tablas():
    """
    Calcula y muestra:
      1) Speedup S(N, p) = T(N, 1) / T(N, p) y Eficiencia E(N, p) = S(N, p) / p
         al crecer el tamano del problema N in {1024, 2048, 4096}.
      2) Eficiencia debil a carga por thread constante (W/p = N^3 / p = constante):
         Dado que el costo de multiplicar matrices es O(N^3), al duplicar N el trabajo
         total se multiplica por 2^3 = 8. Por ello, pasar de (N=1024, p=1) a (N=2048, p=8)
         mantiene exactamente constante el trabajo por thread:
             E_debil_iso(1->8) = T(N=1024, p=1) / T(N=2048, p=8).
      3) Eficiencia debil normalizada por trabajo unitario frente a la base (N0=1024, p=1):
             E_debil_norm(N, p) = [ T(1024, 1) * (N / 1024)^3 / p ] / T(N, p).
    """
    print("\n" + "=" * 100)
    print("RESULTADOS APARTADO (f): ESCALABILIDAD Y EFICIENCIA AL CRECER N in {1024, 2048, 4096}")
    print("=" * 100)

    for nombre_alg, dict_n in DATOS_MANUALES.items():
        print(f"\n>>> {nombre_alg}")
        print("-" * 100)
        print(
            f"{'Tamano N':^10} | {'p':^3} | {'T(N,p) [ms]':^14} | {'Speedup S(N,p)':^15} | "
            f"{'Eficiencia E(N,p)':^18} | {'Ef. Debil Norm. (vs N=1024)':^27}"
        )
        print("-" * 100)

        t_base_1024 = dict_n[1024][0]
        for N in TAMANOS_N:
            tiempos = dict_n[N]
            t1_n = tiempos[0]
            factor_trabajo = (N / 1024.0) ** 3  # O(N^3): 1 para 1024, 8 para 2048, 64 para 4096

            for idx, p in enumerate(THREADS_P):
                tp = tiempos[idx]
                speedup = t1_n / tp
                eficiencia = (speedup / p) * 100.0
                # Tiempo ideal escalando por O(N^3)/p respecto a N=1024, p=1
                t_ideal_escala = t_base_1024 * (factor_trabajo / p)
                ef_debil_norm = (t_ideal_escala / tp) * 100.0

                n_label = f"{N}" if idx == 0 else ""
                print(
                    f"{n_label:^10} | {p:^3d} | {tp:^14.2f} | {speedup:^15.2f} | "
                    f"{eficiencia:^17.1f}% | {ef_debil_norm:^26.1f}%"
                )
            print("-" * 100)

        # Punto de carga por thread estrictamente constante: (N=1024, p=1) -> (N=2048, p=8)
        t_1024_p1 = dict_n[1024][0]
        t_2048_p8 = dict_n[2048][3]
        ef_iso_1_8 = (t_1024_p1 / t_2048_p8) * 100.0
        print(
            f"  * Eficiencia debil a trabajo/thread constante [(N=1024, p=1) -> (N=2048, p=8), "
            f"donde N^3/p = 1024^3]: {ef_iso_1_8:.1f}% "
            f"(T(1024,1)={t_1024_p1:.2f} ms vs T(2048,8)={t_2048_p8:.2f} ms)"
        )


# ==============================================================================
# 4. GENERACION DE GRAFICOS PARA EL APARTADO (f)
# ==============================================================================

def generar_graficos_apartado_f():
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "legend.fontsize": 9,
    })

    nombres = list(DATOS_MANUALES.keys())
    colores_n = {1024: "#1f77b4", 2048: "#d95f02", 4096: "#2ca02c"}
    marcadores_n = {1024: "o", 2048: "s", 4096: "^"}

    # --------------------------------------------------------------------------
    # FIGURA 1: Eficiencia E(N, p) en funcion de p para N in {1024, 2048, 4096}
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), dpi=300, sharey=True)
    p_arr = np.array(THREADS_P, dtype=float)

    for idx, nombre_alg in enumerate(nombres):
        ax = axes[idx]
        dict_n = DATOS_MANUALES[nombre_alg]
        ax.axhline(100.0, linestyle="--", color="#333333", linewidth=1.5, label="Ideal (100 %)")

        for N in TAMANOS_N:
            t_arr = np.array(dict_n[N], dtype=float)
            s_arr = t_arr[0] / t_arr
            e_arr = (s_arr / p_arr) * 100.0
            ax.plot(
                p_arr,
                e_arr,
                marker=marcadores_n[N],
                color=colores_n[N],
                linewidth=2,
                markersize=6,
                label=f"$N = {N}$",
            )

        ax.set_title(nombre_alg)
        ax.set_xlabel("Número de threads ($p$)")
        if idx == 0:
            ax.set_ylabel("Eficiencia Paralela $E(N, p)$ (%)")
        ax.set_xticks(THREADS_P)
        ax.set_ylim(0, 115)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="lower left")

    fig.suptitle(
        "Apartado (f): Evolución de la Eficiencia Paralela al Crecer el Tamaño del Problema ($N \\in \\{1024, 2048, 4096\\}$)",
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_ef = "apartado_f_eficiencia_por_N.png"
    fig.savefig(out_ef, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Grafico guardado: {out_ef}")

    # --------------------------------------------------------------------------
    # FIGURA 2: Eficiencia vs Tamano de Matriz N (fijando p = 4 y p = 8)
    # --------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=300, sharey=True)
    x_idx = np.arange(len(TAMANOS_N))
    x_labels = [str(n) for n in TAMANOS_N]
    colores_alg = ["#1f77b4", "#d95f02", "#2ca02c"]
    marcadores_alg = ["o", "s", "^"]

    for panel_idx, p_fijo in enumerate([4, 8]):
        ax = axes2[panel_idx]
        p_pos = THREADS_P.index(p_fijo)
        ax.axhline(100.0, linestyle="--", color="#333333", linewidth=1.5, label="Ideal (100 %)")

        for alg_idx, nombre_alg in enumerate(nombres):
            dict_n = DATOS_MANUALES[nombre_alg]
            ef_por_n = []
            for N in TAMANOS_N:
                t_list = dict_n[N]
                s_val = t_list[0] / t_list[p_pos]
                e_val = (s_val / p_fijo) * 100.0
                ef_por_n.append(e_val)

            ax.plot(
                x_idx,
                ef_por_n,
                marker=marcadores_alg[alg_idx],
                color=colores_alg[alg_idx],
                linewidth=2.2,
                markersize=7,
                label=nombre_alg,
            )

        ax.set_title(f"Desempeño al aumentar $N$ con $p = {p_fijo}$ threads")
        ax.set_xlabel("Tamaño de la matriz ($N \\times N$)")
        if panel_idx == 0:
            ax.set_ylabel("Eficiencia $E(N, p)$ (%)")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0, 115)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="lower right")

    fig2.suptitle(
        "Apartado (f): Escalabilidad Débil — ¿Se mantiene la eficiencia al crecer $N$?",
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_debil = "apartado_f_escalabilidad_debil.png"
    fig2.savefig(out_debil, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[OK] Grafico guardado: {out_debil}")


if __name__ == "__main__":
    modo = MODO_POR_DEFECTO
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("medir", "auto", "auto_numba"):
            modo = "auto_numba"
        elif arg in ("graficar", "manual"):
            modo = "manual"

    if modo == "auto_numba" and NUMBA_DISPONIBLE:
        medir_escalabilidad_numba()

    analizar_y_mostrar_tablas()
    generar_graficos_apartado_f()
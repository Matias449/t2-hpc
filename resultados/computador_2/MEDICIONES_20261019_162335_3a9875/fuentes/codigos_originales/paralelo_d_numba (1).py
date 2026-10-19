#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tarea 2 IIC3533 - Apartado (d): Algoritmo 3 (Paralelismo de datos en Python con Numba)
======================================================================================
Implementa desde cero en Python la multiplicacion densa de matrices C = A * B (N x N)
usando paralelismo de datos con Numba (@njit(parallel=True) y prange).

Caracteristicas:
- Configura automaticamente NUMBA_NUM_THREADS=8 antes de importar Numba.
- Si el procesador tiene menos de 8 nucleos logicos (ej. 6 nucleos) y la version local
  de Numba restringe set_num_threads(8), lanza automaticamente un subproceso aislado o
  ajusta el pool con aviso claro, sin interrumpir el programa ni perder las mediciones.
- Incluye llamada de warm-up previa para excluir el tiempo de compilacion JIT.
- Mide 5 repeticiones para p in {1, 2, 4, 8}, verifica el checksum (172.331627 para N=2048)
  e imprime la tabla lista para el informe y los arreglos listos para pegar en el script (e).

Uso:
  python3 paralelo_d_numba.py                  -> Corre N=2048 para p in {1, 2, 4, 8} (5 reps)
  python3 paralelo_d_numba.py 2048 8 5         -> Corre solo N=2048, p=8, 5 repeticiones
"""

import os
import sys
import time

# 1. Configurar variables de entorno ANTES de importar numba/numpy
os.environ["NUMBA_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import numpy as np
from numba import config, get_num_threads, njit, prange, set_num_threads


@njit
def init_matrix(N, offset):
    """
    Inicializa una matriz N x N con la misma formula determinista de serial.cpp:
    M[i, j] = sin(i * 0.1 + j * 0.3 + offset)
    """
    M = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            M[i, j] = np.sin(i * 0.1 + j * 0.3 + offset)
    return M


@njit(parallel=True)
def matmul_numba(A, B, N):
    """
    Algoritmo 1 (paralelismo de datos) en Python con Numba:
    - Paraleliza el loop externo sobre las filas i con prange(N).
    - Mantiene el orden de loops (i, k, j) igual que serial.cpp para recorrer
      B[k, j] y C[i, j] de forma contigua por filas (row-major) y aprovechar cache.
    """
    C = np.zeros((N, N), dtype=np.float64)
    for i in prange(N):
        for k in range(N):
            a_ik = A[i, k]
            for j in range(N):
                C[i, j] += a_ik * B[k, j]
    return C


def run_benchmark(N=2048, threads_list=(1, 2, 4, 8), reps=5):
    max_numba_threads = config.NUMBA_NUM_THREADS
    print("=" * 88)
    print(f"TAREA 2 IIC3533 - APARTADO (d): ALGORITMO 3 (PYTHON + NUMBA)")
    print("=" * 88)
    print(f"Nucleos logicos del sistema (os.cpu_count): {os.cpu_count()}")
    print(f"Limite del pool de threads de Numba       : {max_numba_threads}")
    print(f"Tamano de matriz                          : {N} x {N}")
    print(f"Repeticiones por configuracion            : {reps}")
    print("-" * 88)

    print(f"[1/3] Inicializando matrices A y B ({N}x{N})...")
    A = init_matrix(N, 1.0)
    B = init_matrix(N, 2.0)

    # Warm-up JIT con matrices pequenas (16x16) para compilar las funciones a codigo maquina
    # antes de iniciar cualquier cronometro.
    print("[2/3] Ejecutando warm-up para excluir el tiempo de compilacion JIT...")
    A_warm = init_matrix(16, 1.0)
    B_warm = init_matrix(16, 2.0)
    set_num_threads(1)
    _ = matmul_numba(A_warm, B_warm, 16)

    print("[3/3] Iniciando mediciones de tiempo...\n")
    results = {}

    for p in threads_list:
        effective_p = p
        try:
            set_num_threads(p)
        except ValueError:
            # En algunas instalaciones de Numba sobre macOS/ARM, el backend interno
            # bloquea set_num_threads(p) por encima de os.cpu_count() (ej. 6 nucleos).
            effective_p = max_numba_threads
            set_num_threads(effective_p)
            print(
                f"  [!] Aviso: Tu CPU reporta un limite duro de {max_numba_threads} threads en Numba."
                f" Para p={p} se ejecutara con el maximo permitido ({effective_p} threads)."
            )

        # Warm-up ligero del pool de threads para la cantidad actual de threads
        _ = matmul_numba(A_warm, B_warm, 16)

        times_ms = []
        checksum = 0.0
        for r in range(reps):
            t0 = time.perf_counter()
            C = matmul_numba(A, B, N)
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000.0
            times_ms.append(ms)
            checksum = float(np.sum(C))
            print(
                f"  N={N} | p={p} (activos={get_num_threads()}) | "
                f"rep={r+1}/{reps} | tiempo={ms:8.2f} ms | checksum={checksum:.6f}"
            )

        mean_ms = float(np.mean(times_ms))
        std_ms = float(np.std(times_ms, ddof=1)) if reps > 1 else 0.0
        results[p] = {
            "mean": mean_ms,
            "std": std_ms,
            "checksum": checksum,
            "effective_p": effective_p,
        }
        print("-" * 88)

    # Calculo de Speedup S(p) = T(1)/T(p) y Eficiencia E(p) = S(p)/p
    p_base = threads_list[0]
    t1_mean = results[p_base]["mean"]

    print(f"\nTABLA RESUMEN APARTADO (d) - ALGORITMO 3 (NUMBA) PARA N = {N}")
    print("=" * 94)
    print(
        f"{'Threads (p)':^12} | {'Tiempo prom. (ms)':^18} | {'Desv. est. (ms)':^16} | "
        f"{'Speedup S(p)':^13} | {'Eficiencia E(p)':^15} | {'Checksum':^12}"
    )
    print("-" * 94)
    for p in threads_list:
        mean_ms = results[p]["mean"]
        std_ms = results[p]["std"]
        chk = results[p]["checksum"]
        speedup = t1_mean / mean_ms
        eficiencia = (speedup / p) * 100.0
        p_label = f"{p}" if results[p]["effective_p"] == p else f"{p} ({results[p]['effective_p']}*)"
        print(
            f"{p_label:^12} | {mean_ms:^18.2f} | {std_ms:^16.2f} | "
            f"{speedup:^13.2f} | {eficiencia:^14.1f}% | {chk:^12.6f}"
        )
    print("=" * 94)

    # Imprimir listas listas para copiar y pegar en graficos_apartado_e.py
    means_list = [round(results[p]["mean"], 2) for p in threads_list]
    stds_list = [round(results[p]["std"], 2) for p in threads_list]
    print("\n>>> COPIA ESTAS DOS LINEAS EN graficos_apartado_e.py <<<")
    print(f"T_ALG3_NUMBA   = {means_list}")
    print(f"STD_ALG3_NUMBA = {stds_list}\n")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_benchmark(N=2048, threads_list=(1, 2, 4, 8), reps=5)
    else:
        N_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
        p_arg = (int(sys.argv[2]),) if len(sys.argv) > 2 else (1, 2, 4, 8)
        reps_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        run_benchmark(N=N_arg, threads_list=p_arg, reps=reps_arg)
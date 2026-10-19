"""Apartado d con el mismo alcance de tiempo que C++: solo multiplicación."""
import os
os.environ["NUMBA_NUM_THREADS"] = "8"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
import time
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads


@njit(cache=True)
def inicializar(N, offset):
    M = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            M[i, j] = np.sin(i * 0.1 + j * 0.3 + offset)
    return M


@njit(parallel=True, cache=True)
def multiplicar(A, B, C):
    N = C.shape[0]
    for i in prange(N):
        for k in range(N):
            a = A[i, k]
            for j in range(N):
                C[i, j] += a * B[k, j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("N", type=int)
    ap.add_argument("n", type=int)  # interfaz común; algoritmo directo ignora n
    ap.add_argument("p", type=int)
    ap.add_argument("reps", type=int)
    ap.add_argument("dump", nargs="?")
    args = ap.parse_args()
    if args.N not in [2**i for i in range(13)] or args.p not in (1, 2, 4, 8) or args.reps < 1:
        ap.error("N potencia de dos <=4096, p=1,2,4,8 y reps>=1")
    set_num_threads(args.p)
    if get_num_threads() != args.p:
        raise RuntimeError("Threads efectivos diferentes de los solicitados")
    A, B = inicializar(args.N, 1.0), inicializar(args.N, 2.0)
    C = np.zeros_like(A)
    multiplicar(A, B, C)  # JIT y warm-up completo fuera del cronómetro
    for r in range(1, args.reps + 1):
        C.fill(0)
        t0 = time.perf_counter()
        multiplicar(A, B, C)
        ms = (time.perf_counter() - t0) * 1000
        if not np.isfinite(C).all():
            raise RuntimeError("Resultado no finito")
        print(json.dumps(dict(rep=r, ms=ms, checksum=float(C.sum()),
                              threads=get_num_threads())), flush=True)
    if args.dump:
        np.savetxt(args.dump, C.reshape(-1), fmt="%.17g")


if __name__ == "__main__":
    main()

"""Planifica y guarda experimentos sin usar datos manuales ni sobrescribir corridas."""
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
import numpy as np
from compilar import HERE, ROOT, SRC, sha

THREADS = [1, 2, 4, 8]
BLOCKS = [2048, 1024, 512, 256, 128, 64, 32]


def plan(variante):
    jobs = [("serial", 2048, n, 1) for n in BLOCKS]
    for N in [1024, 2048, 4096]:
        for name in ["datos", variante, "numba"]:
            for p in THREADS:
                jobs.append((name, N, 64 if name == variante else N, p))
    for n in BLOCKS:
        for p in THREADS:
            job = (variante, 2048, n, p)
            if job not in jobs:
                jobs.append(job)
    return jobs


def checksum_referencia(N):
    # Sum(C)=sum_k sum_i A[i,k] * sum_j B[k,j]. Identidades trigonométricas
    # dan una referencia O(N), independiente del kernel cúbico y sin BLAS.
    x = np.arange(N, dtype=np.float64)
    sa, ca = np.sin(x*.1).sum(), np.cos(x*.1).sum()
    sb, cb = np.sin(x*.3).sum(), np.cos(x*.3).sum()
    col_A = sa*np.cos(x*.3+1) + ca*np.sin(x*.3+1)
    row_B = sb*np.cos(x*.1+2) + cb*np.sin(x*.1+2)
    return float(np.sum(col_A*row_B))


def capture(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return (r.stdout.strip() or None) if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def hardware():
    info = dict(system=platform.platform(), architecture=platform.machine(),
                python=sys.version, logical_cpus=os.cpu_count(),
                packages={n: importlib.metadata.version(n) for n in ["numpy", "numba", "llvmlite", "matplotlib"]})
    if sys.platform == "darwin":
        keys = ["hw.model", "machdep.cpu.brand_string", "hw.physicalcpu", "hw.logicalcpu",
                "hw.memsize", "hw.cpufrequency", "hw.cpufrequency_max", "hw.nperflevels",
                "hw.perflevel0.name", "hw.perflevel0.physicalcpu", "hw.perflevel0.cpufrequency_max",
                "hw.perflevel1.name", "hw.perflevel1.physicalcpu", "hw.perflevel1.cpufrequency_max"]
        info["sysctl"] = {k: capture(["sysctl", "-n", k]) for k in keys}
        info["power"] = capture(["pmset", "-g", "batt"])
    else:
        info["lscpu"] = capture(["lscpu", "-J"])
    info["clock_note"] = "null significa que el sistema no expone la frecuencia; no es una medición del reloj"
    return info


def check_build():
    info = json.loads((HERE / "compilacion.json").read_text())
    for rel, expected in info["sources"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"Fuente cambiada: {rel}. Ejecutar preparar.command nuevamente.")
    for name, expected in info["binaries"].items():
        if sha(HERE / "bin" / name) != expected:
            raise RuntimeError(f"Binario cambiado: {name}. Recompilar.")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true", help="Mostrar plan sin ejecutar nada")
    ap.add_argument("--variante", choices=["cuadrantes", "tareas"], default="cuadrantes")
    ap.add_argument("--repeticiones", type=int, default=5)
    ap.add_argument("--equipo", default=platform.machine()+"-"+platform.system())
    ap.add_argument("--prueba", action="store_true", help="Solo N=32; resultados separados de entrega")
    args = ap.parse_args()
    if args.repeticiones < 2:
        ap.error("Usar al menos dos repeticiones para calcular desviación estándar")
    jobs = plan(args.variante)
    if args.prueba:
        jobs = [(name, 32, 8 if name == args.variante else 32, p)
                for name in ["serial", "datos", args.variante, "numba"]
                for p in ([1] if name == "serial" else THREADS)]
    if args.plan:
        print(f"Variante c: {args.variante}; {len(jobs)} configuraciones; "
              f"{len(jobs)*args.repeticiones} repeticiones medidas; un warm-up completo por configuración.")
        for name, N, n, p in jobs:
            print(f"  {name:12s} N={N:4d} n={n:4d} p={p}")
        return
    build = check_build()
    # Evita que dos procesos de esta carpeta compitan y contaminen las mediciones.
    import fcntl
    with (HERE / ".ejecucion.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Ya hay una ejecución activa en esta carpeta")
        run(args, jobs, build)


def run(args, jobs, build):
    stamp = datetime.now(ZoneInfo("America/Santiago")).strftime("%Y%m%d_%H%M%S")
    prefix = "PRUEBA" if args.prueba else "MEDICIONES"
    folder = ROOT / "resultados" / f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}"
    folder.mkdir(parents=True)
    snapshot = folder / "fuentes"
    shutil.copytree(HERE, snapshot / "ejecucion", ignore=shutil.ignore_patterns("bin", "__pycache__", ".ejecucion.lock"))
    shutil.copytree(SRC, snapshot / "codigos_originales")
    refs = {N: checksum_referencia(N) for N in sorted({job[1] for job in jobs})}
    meta = dict(status="running", started=datetime.now(timezone.utc).isoformat(),
                purpose="prueba pequeña" if args.prueba else "mediciones para la tarea",
                equipo=args.equipo, arguments=vars(args), hardware=hardware(), build=build,
                expected_checksums=refs, jobs=jobs,
                timing="Solo kernel; reserva, inicialización, cero, checksum y JIT fuera. Warm-up completo por configuración.",
                validation="Matrices completas pequeñas y checksum independiente en cada repetición; no valida cada elemento de matrices grandes.")
    meta_path = folder / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("Guardando en:", folder, flush=True)
    env = dict(os.environ, NUMBA_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OMP_DYNAMIC="FALSE", MPLBACKEND="Agg")
    proc = None
    try:
        print("Validando matrices pequeñas antes de medir...", flush=True)
        validation = subprocess.run([sys.executable, str(HERE / "validar.py")], env=env,
                                    text=True, capture_output=True)
        (folder / "validacion.log").write_text(validation.stdout + validation.stderr)
        validation.check_returncode()
        shutil.copy2(HERE / "validacion.json", folder / "validacion.json")
        fields = ["equipo", "algorithm", "N", "n", "p", "team_p", "rep", "ms", "checksum", "expected_checksum"]
        with (folder / "mediciones.csv").open("w", newline="") as raw, (folder / "ejecucion.log").open("w") as log:
            writer = csv.DictWriter(raw, fieldnames=fields)
            writer.writeheader()
            raw.flush()
            for idx, (name, N, n, p) in enumerate(jobs, 1):
                base = [sys.executable, str(HERE / "numba_kernel.py")] if name == "numba" else [str(HERE / "bin" / name)]
                cmd = [*base, str(N), str(n), str(p), str(args.repeticiones)]
                label = f"[{idx}/{len(jobs)}] {name} N={N} n={n} p={p}"
                print(label, flush=True)
                log.write(label + "\n" + json.dumps(cmd) + "\n")
                log.flush()
                with (folder / "stderr.log").open("a") as err:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, env=env)
                    count = 0
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        row = json.loads(line)
                        count += 1
                        if row["rep"] != count or row["threads"] != p:
                            raise RuntimeError("Repetición o threads inesperados")
                        if not np.isfinite(row["ms"]) or row["ms"] <= 0:
                            raise RuntimeError("Tiempo inválido")
                        if not np.isclose(row["checksum"], refs[N], rtol=1e-7, atol=1e-5):
                            raise RuntimeError(f"Checksum incorrecto en {label}: {row['checksum']} frente a {refs[N]}")
                        writer.writerow(dict(equipo=args.equipo, algorithm=name, N=N, n=n, p=p,
                                             team_p=row["threads"], rep=row["rep"], ms=row["ms"],
                                             checksum=row["checksum"], expected_checksum=refs[N]))
                        raw.flush()
                        os.fsync(raw.fileno())
                        print(f"  repetición {count}: {row['ms']:.3f} ms", flush=True)
                    if proc.wait() != 0 or count != args.repeticiones:
                        raise RuntimeError(f"Falló {label}; revisar stderr.log")
                    proc = None
        subprocess.run([sys.executable, str(HERE / "analizar.py"), str(folder)], check=True, env=env)
        meta["status"] = "complete"
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        meta["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        meta["error"] = repr(exc)
        raise
    finally:
        meta["finished"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("Completado y guardado:", folder, flush=True)


if __name__ == "__main__":
    main()

"""Pruebas pequeñas de matrices completas, sin mediciones para el informe."""
from pathlib import Path
import json
import subprocess
import tempfile
import numpy as np
from numba_kernel import inicializar, multiplicar
from numba import set_num_threads
from compilar import HERE


def validar():
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp) / "matriz.txt"
        for N in [1, 16, 64]:
            A, B = inicializar(N, 1.0), inicializar(N, 2.0)
            ref = A @ B  # referencia independiente; nunca se usa en benchmarks
            for name in ["serial", "datos", "tareas", "cuadrantes"]:
                for n in sorted({1, max(1, N // 4), N}):
                    for p in ([1] if name == "serial" else [1, 2, 4, 8]):
                        output = subprocess.check_output(
                            [str(HERE / "bin" / name), str(N), str(n), str(p), "1", str(dump)],
                            text=True)
                        row = json.loads(output.strip())
                        got = np.loadtxt(dump).reshape(N, N)
                        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)
                        assert row["threads"] == p
                        checks.append(dict(algorithm=name, N=N, n=n, p=p,
                                           max_abs_error=float(np.max(np.abs(got-ref)))))
            for p in [1, 2, 4, 8]:
                set_num_threads(p)
                got = np.zeros_like(A)
                multiplicar(A, B, got)
                np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)
                checks.append(dict(algorithm="numba", N=N, n=N, p=p,
                                   max_abs_error=float(np.max(np.abs(got-ref)))))
    return checks


def main():
    from datetime import datetime, timezone
    checks = validar()
    report = dict(date=datetime.now(timezone.utc).isoformat(), status="passed",
                  scope="Matrices pequeñas completas; no son benchmarks de entrega", checks=checks)
    (HERE / "validacion.json").write_text(json.dumps(report, indent=2))
    print(f"VALIDACIÓN CORRECTA: {len(checks)} comparaciones de matrices completas")


if __name__ == "__main__":
    main()

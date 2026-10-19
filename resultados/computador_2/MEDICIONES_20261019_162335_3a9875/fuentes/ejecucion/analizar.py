"""Produce tablas y gráficos exclusivamente a partir de mediciones.csv."""
from pathlib import Path
import argparse
import csv
import json
import statistics
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = dict(datos="OpenMP datos", tareas="OpenMP tareas", cuadrantes="OpenMP cuadrantes", numba="Numba", serial="Serial")


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analizar(folder):
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text())
    with (folder / "mediciones.csv").open() as f:
        raw = list(csv.DictReader(f))
    groups = defaultdict(list)
    for row in raw:
        key = (row["algorithm"], int(row["N"]), int(row["n"]), int(row["p"]))
        groups[key].append(float(row["ms"]))
    expected = {tuple(job) for job in meta["jobs"]}
    reps = meta["arguments"]["repeticiones"]
    if set(groups) != expected or any(len(v) != reps for v in groups.values()):
        raise RuntimeError("Datos incompletos: no se generarán gráficos de una corrida parcial")
    means = {key: statistics.mean(v) for key, v in groups.items()}
    summary = []
    for (alg, N, n, p), values in sorted(groups.items()):
        base = means[(alg, N, n, 1)]
        mean = statistics.mean(values)
        summary.append(dict(algorithm=alg, N=N, n=n, p=p, repetitions=len(values),
                            mean_ms=mean, std_ms=statistics.stdev(values),
                            T1_ms=base, speedup=base/mean, efficiency=base/mean/p))
    save_csv(folder / "resumen.csv", summary)
    figs = folder / "graficos"
    figs.mkdir(exist_ok=True)
    test = meta["arguments"]["prueba"]
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .25})
    N_main = 32 if test else 2048
    main_n = 8 if test else 64
    algorithms = ["datos", meta["arguments"]["variante"], "numba"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for alg in algorithms:
        rows = [r for r in summary if r["algorithm"] == alg and r["N"] == N_main
                and r["n"] == (main_n if alg in ("tareas", "cuadrantes") else N_main)]
        rows.sort(key=lambda r: r["p"])
        p = [r["p"] for r in rows]
        line = axes[0].errorbar(p, [r["mean_ms"] for r in rows], yerr=[r["std_ms"] for r in rows],
                               marker="o", capsize=3, label=LABELS[alg])
        axes[0].plot(p, [rows[0]["T1_ms"]/x for x in p], "--", alpha=.6,
                     color=line.lines[0].get_color(), label="Ideal " + LABELS[alg])
        axes[1].plot(p, [r["speedup"] for r in rows], "o-", label=LABELS[alg])
        axes[2].plot(p, [100*r["efficiency"] for r in rows], "o-", label=LABELS[alg])
    axes[1].plot([1, 2, 4, 8], [1, 2, 4, 8], "k--", label="Ideal S=p")
    axes[2].axhline(100, color="black", linestyle="--", label="Ideal 100 %")
    for ax, title, ylabel in zip(axes, ["Tiempo", "Speedup", "Eficiencia"], ["ms", "T(1)/T(p)", "%"]):
        ax.set(title=title, xlabel="Threads solicitados p", ylabel=ylabel, xticks=[1, 2, 4, 8])
        ax.legend(fontsize=7)
    title = "PRUEBA PEQUEÑA — " if test else ""
    fig.suptitle(f"{title}N={N_main} — T(1) propio de cada algoritmo — {meta['equipo']}")
    fig.tight_layout()
    fig.savefig(figs / "comparacion.png", dpi=200)
    plt.close(fig)
    if test:
        (folder / "LEEME.txt").write_text("Solo prueba funcional N=32. No usar estos tiempos en el informe.\n")
        return
    # a y e: efecto del bloque, conservando series independientes por p.
    fig, ax = plt.subplots(figsize=(9, 5))
    serial = sorted([r for r in summary if r["algorithm"] == "serial"], key=lambda r: r["n"])
    ax.errorbar([r["n"] for r in serial], [r["mean_ms"] for r in serial],
                yerr=[r["std_ms"] for r in serial], marker="o", capsize=3, label="Serial")
    for p in [1, 2, 4, 8]:
        rows = sorted([r for r in summary if r["algorithm"] == algorithms[1] and r["N"] == 2048 and r["p"] == p], key=lambda r: r["n"])
        ax.errorbar([r["n"] for r in rows], [r["mean_ms"] for r in rows],
                    yerr=[r["std_ms"] for r in rows], marker="o", capsize=3, label=f"{LABELS[algorithms[1]]}, p={p}")
    ax.set_xscale("log", base=2)
    ax.set(xticks=sorted([r["n"] for r in serial]), xlabel="Tamaño de bloque n", ylabel="Tiempo promedio (ms)", title="Efecto de n — N=2048")
    ax.set_xticklabels([str(r["n"]) for r in serial])
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "bloques.png", dpi=200)
    plt.close(fig)
    # f: escalabilidad fuerte para cada N, rotulada explícitamente.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, alg in zip(axes, algorithms):
        for N in [1024, 2048, 4096]:
            rows = sorted([r for r in summary if r["algorithm"] == alg and r["N"] == N
                           and r["n"] == (64 if alg in ("tareas", "cuadrantes") else N)], key=lambda r: r["p"])
            ax.plot([r["p"] for r in rows], [100*r["efficiency"] for r in rows], "o-", label=f"N={N}")
        ax.axhline(100, linestyle="--", color="black")
        ax.set(title=LABELS[alg], xlabel="Threads", ylabel="Eficiencia fuerte (%)", xticks=[1, 2, 4, 8])
        ax.legend()
    fig.suptitle("Eficiencia fuerte a distintos tamaños; complemento del estudio débil")
    fig.tight_layout()
    fig.savefig(figs / "eficiencia_por_tamano.png", dpi=200)
    plt.close(fig)
    weak = []
    for alg in algorithms:
        for small, large in [(1024, 2048), (2048, 4096)]:
            n0 = 64 if alg in ("tareas", "cuadrantes") else small
            n1 = 64 if alg in ("tareas", "cuadrantes") else large
            t0, t1 = means[(alg, small, n0, 1)], means[(alg, large, n1, 8)]
            weak.append(dict(algorithm=alg, N_base=small, p_base=1, N_final=large, p_final=8,
                             base_ms=t0, final_ms=t1, weak_efficiency=t0/t1))
    save_csv(folder / "escalabilidad_debil.csv", weak)
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, (small, large) in enumerate([(1024, 2048), (2048, 4096)]):
        vals = [next(r["weak_efficiency"]*100 for r in weak if r["algorithm"] == alg and r["N_base"] == small) for alg in algorithms]
        ax.bar([x + offset*.36 for x in range(3)], vals, width=.36, label=f"({small},1) → ({large},8)")
    ax.axhline(100, color="black", linestyle="--", label="Ideal 100 %")
    ax.set(xticks=[x+.18 for x in range(3)], xticklabels=[LABELS[a] for a in algorithms],
           ylabel="Eficiencia débil (%)", title="Carga constante N³/p; dos pares independientes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "escalabilidad_debil.png", dpi=200)
    plt.close(fig)
    (folder / "LEEME.txt").write_text(
        "Resultados exclusivos de este computador.\n"
        "mediciones.csv: repeticiones crudas en ms. resumen.csv: media, desviación muestral, T(1), S y E.\n"
        "Cada serie usa su T(1) con el mismo algoritmo, N y n. La variante de c queda en metadata.json.\n"
        "Para justificar n, consultar bloques.png y el barrido completo en resumen.csv.\n"
        "Escalabilidad débil: N³/p constante; se muestran dos pares independientes, no una trayectoria de 3 tamaños.\n"
        "El apartado g requiere resultados del otro computador, aún no incluidos aquí.\n"
        "metadata.json guarda entorno, compilación y estado; fuentes/ conserva los códigos utilizados.\n")
    print("Tablas y cuatro figuras guardadas en", folder)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta", type=Path)
    analizar(ap.parse_args().carpeta)

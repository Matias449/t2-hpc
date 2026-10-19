# Tarea 2: Multiplicación de Matrices con OpenMP y Numba
**IIC3533 · Computación de Alto Rendimiento (HPC)**  
*Pontificia Universidad Católica de Chile*

Repositorio automatizado y reproducible para el desarrollo experimental de la Tarea 2. Compara el rendimiento, escalabilidad fuerte/débil y eficiencia de tres algoritmos de multiplicación densa de matrices $C = A \cdot B$ de tamaño $N \times N$:
1. **Algoritmo 1 (C++ con OpenMP):** Paralelismo de datos por filas (`src/paralelo_b.cpp`).
2. **Algoritmo 2 (C++ con OpenMP):** Dividir y conquistar recursivo por cuadrantes (`src/paralelo_c.cpp`).
3. **Algoritmo 3 (Python con Numba):** Paralelismo de datos compilado JIT nativo (`src/paralelo_d_numba.py`).

---

## 🚀 Inicio Rápido (Quickstart para Compañeros de Grupo)

Si acabas de clonar este repositorio en tu computador (**Computador 2**), sigue estos sencillos pasos:

### 1. Instalar dependencias de Python
```bash
pip install -r requirements.txt
```

### 2. Verificar compilador con soporte OpenMP
- **En Linux (Ubuntu / Debian):**
  ```bash
  sudo apt-get update && sudo apt-get install -y build-essential libomp-dev
  ```
- **En macOS:**
  ```bash
  brew install libomp
  ```
- **En Windows:** Asegurar `g++` con soporte OpenMP (MinGW-w64).

### 3. Ejecutar todos los experimentos automáticamente
Desde la raíz del repositorio, ejecuta:
```bash
python3 ejecutar.py --computador 2 --pdf
```
*(O también: `python3 scripts/ejecutar_experimentos.py --computador 2 --pdf`)*

Este comando único se encarga automáticamente de:
1. Detectar los núcleos, frecuencia, RAM y arquitectura de tu procesador.
2. Compilar los códigos fuente C++ con optimizaciones máximas (`-O3 -march=native -fopenmp`).
3. Ejecutar en orden todos los apartados (a, b, c, d, e, f, g).
4. Validar la consistencia matemática de los resultados (*checksum* `172.331627`).
5. Generar tus gráficos y tu informe en `resultados/computador_2/INFORME_EXPERIMENTOS_COMPUTADOR_2.md`.
6. Contrastar tus resultados con el **Computador 1 (Apple A18 Pro)** para el **Apartado (g)**.
7. Compilar tu informe a un PDF estático listo para entrega.

---

## 🤖 Para Agentes de Inteligencia Artificial (Codex, Claude Code, Antigravity)

Si estás trabajando con un asistente o agente de IA en tu terminal o IDE, este repositorio incluye instrucciones completas, contexto teórico y comandos pre-configurados en:
👉 **[AGENTS.md](AGENTS.md)**

---

## 📁 Estructura del Repositorio

```text
├── README.md                           # Guía general de uso
├── AGENTS.md                           # Instrucciones para agentes IA (Codex / Claude)
├── requirements.txt                    # Dependencias de Python
├── ejecutar.py                         # Acceso directo para correr experimentos
├── .gitignore                          # Exclusiones de Git
│
├── docs/                               # Documentación y enunciado oficial
│   └── tarea02.pdf                     # Pauta oficial de la Tarea 2
│
├── src/                                # Códigos fuente C++ y Python
│   ├── serial.cpp                      # Referencia serial base y por bloques
│   ├── paralelo_b.cpp                  # Algoritmo 1: OpenMP Paralelismo de datos (filas)
│   ├── paralelo_c.cpp                  # Algoritmo 2: OpenMP Dividir y Conquistar (tareas)
│   ├── paralelo_c_cuadrantes.cpp       # Variante cuadrantes
│   └── paralelo_d_numba.py             # Algoritmo 3: Python con Numba (@njit + prange)
│
├── scripts/                            # Scripts de automatización y benchmarking
│   ├── ejecutar_experimentos.py        # Orquestador universal de benchmarks
│   ├── convertir_informe_a_pdf.py      # Compilador de Markdown a PDF con soporte KaTeX
│   ├── graficos_apartado_e.py          # Generador de gráficos de escalamiento (Apartado e)
│   └── apartado_f_escalabilidad_debil.py # Análisis de escalabilidad débil (Apartado f)
│
└── resultados/                         # Informes, datos brutos y gráficos generados
    ├── computador_1/                   # Resultados del Computador 1 (Apple A18 Pro)
    │   ├── INFORME_EXPERIMENTOS.md     # Informe completo en Markdown
    │   ├── INFORME_EXPERIMENTOS.pdf    # PDF estático de 12 páginas listo para entrega
    │   ├── INFORME_EXPERIMENTOS.html   # Reporte HTML interactivo autocontenido
    │   ├── resultados_computador_1.json # Datos brutos en JSON
    │   └── figuras/                    # Figuras en alta resolución (300 DPI)
    └── computador_2/                   # Destino para mediciones del Computador 2
        └── .gitkeep
```

---

## 📊 Resumen de Apartados Evaluados

- **(a) Base serial y tamaño de bloque:** $N = 2048$, $n \in \{2048, 1024, 512, 256, 128, 64, 32\}$.
- **(b) Algoritmo 1:** OpenMP paralelismo de datos con $p \in \{1, 2, 4, 8\}$ hilos.
- **(c) Algoritmo 2:** OpenMP dividir y conquistar con $p \in \{1, 2, 4, 8\}$ hilos y variación de $n$.
- **(d) Algoritmo 3:** Python con Numba (excluyendo tiempo de compilación JIT vía *warm-up*).
- **(e) Comparación y escalamiento:** Curvas de $T(p)$, Speedup relativo/absoluto y eficiencia vs. curva ideal.
- **(f) Escalabilidad débil:** Evaluación con $N \in \{1024, 2048, 4096\}$ y cálculo de eficiencia débil iso-granular $\mathcal{E}_{débil}^{iso}(1 \to 8)$.
- **(g) Segundo computador:** Comparativa técnica de hardware, heterogeneidad de núcleos y memorias.

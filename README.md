# Tarea 2: Multiplicación de Matrices con OpenMP y Numba
**IIC3533 · Computación de Alto Rendimiento (HPC)**  
*Pontificia Universidad Católica de Chile*

Repositorio automatizado y reproducible para el desarrollo experimental de la Tarea 2. Compara el rendimiento, escalabilidad fuerte/débil y eficiencia de tres algoritmos de multiplicación densa de matrices $C = A \cdot B$ de tamaño $N \times N$:
1. **Algoritmo 1 (C++ con OpenMP):** Paralelismo de datos por filas (`paralelo_b.cpp`).
2. **Algoritmo 2 (C++ con OpenMP):** Dividir y conquistar recursivo por cuadrantes (`paralelo_c.cpp`).
3. **Algoritmo 3 (Python con Numba):** Paralelismo de datos compilado JIT nativo (`paralelo_d_numba.py`).

---

## 🚀 Inicio Rápido (Quickstart para Compañeros de Grupo)

Si acabas de clonar este repositorio en tu computador (**Computador 2**), sigue estos pasos:

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

### 3. Ejecutar todos los experimentos automáticamente
```bash
python3 ejecutar_experimentos.py --computador 2 --pdf
```
Este comando único se encarga de:
- Detectar automáticamente los núcleos, frecuencia, RAM y arquitectura de tu procesador.
- Compilar los códigos fuente C++ con optimizaciones máximas (`-O3 -march=native -fopenmp`).
- Ejecutar en orden todos los apartados (a, b, c, d, e, f, g).
- Validar la consistencia matemática de los resultados (*checksum* `172.331627`).
- Generar tu informe en Markdown: `INFORME_EXPERIMENTOS_COMPUTADOR_2.md`.
- Generar la tabla comparativa directa contra el Computador 1 (Apple A18 Pro).
- Compilar el reporte final a un PDF estático listo para entrega.

---

## 🤖 Para Agentes de Inteligencia Artificial (Codex, Claude Code, Antigravity)

Si estás trabajando con un asistente o agente de IA en tu terminal o IDE, este repositorio incluye instrucciones completas, contexto teórico y comandos pre-configurados en:
👉 **[AGENTS.md](AGENTS.md)**

---

## 📁 Estructura del Repositorio

| Archivo | Descripción |
| :--- | :--- |
| **`tarea02.pdf`** | Enunciado oficial de la tarea y pauta de evaluación. |
| **`requirements.txt`** | Dependencias de Python (`numpy`, `matplotlib`, `numba`, `markdown`). |
| **`serial.cpp`** | Código de referencia secuencial base y partición por bloques. |
| **`paralelo_b.cpp`** | Algoritmo 1: OpenMP Paralelismo de datos (`#pragma omp parallel for`). |
| **`paralelo_c.cpp`** | Algoritmo 2: OpenMP Dividir y Conquistar (`#pragma omp task` por cuadrantes). |
| **`paralelo_d_numba.py`** | Algoritmo 3: Python con Numba (`@njit(parallel=True)` + `prange`). |
| **`ejecutar_experimentos.py`** | **Orquestador principal:** compila, mide repeticiones y genera los informes. |
| **`convertir_informe_a_pdf.py`** | Compilador de Markdown a PDF estático con soporte KaTeX para fórmulas. |
| **`INFORME_EXPERIMENTOS.md`** | Informe completo de resultados medidos en el **Computador 1 (Apple A18 Pro)**. |
| **`INFORME_EXPERIMENTOS.pdf`** | PDF de 12 páginas listo para entrega con resultados del **Computador 1**. |
| **`resultados_computador_1.json`** | Datos experimentales brutos del Computador 1 para contrastar en Apartado (g). |

---

## 📊 Apartados Evaluados en la Pauta

- **(a) Base serial y tamaño de bloque:** $N = 2048$, $n \in \{2048, 1024, 512, 256, 128, 64, 32\}$.
- **(b) Algoritmo 1:** OpenMP paralelismo de datos con $p \in \{1, 2, 4, 8\}$ hilos.
- **(c) Algoritmo 2:** OpenMP dividir y conquistar con $p \in \{1, 2, 4, 8\}$ hilos y variación de $n$.
- **(d) Algoritmo 3:** Python con Numba (excluyendo tiempo de compilación JIT vía *warm-up*).
- **(e) Comparación y escalamiento:** Curvas de $T(p)$, Speedup relativo/absoluto y eficiencia vs. curva ideal.
- **(f) Escalabilidad débil:** Evaluación con $N \in \{1024, 2048, 4096\}$ y cálculo de eficiencia débil iso-granular $\mathcal{E}_{débil}^{iso}(1 \to 8)$.
- **(g) Segundo computador:** Comparativa técnica de hardware, heterogeneidad de núcleos y memorias.

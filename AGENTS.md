# Instrucciones y Contexto para Agentes de IA (Codex, Claude Code, Antigravity)

Este repositorio contiene el código fuente, suite de benchmarks y documentación técnica de la **Tarea 2 de IIC3533 - Computación de Alto Rendimiento (HPC)**, Pontificia Universidad Católica de Chile.

---

## 1. Contexto de la Tarea y Objetivo del Agente

La tarea estudia la multiplicación de matrices densas $C = A \cdot B$ de tamaño $N \times N$ ($\mathcal{O}(N^3)$ operaciones) comparando tres algoritmos paralelos:
1. **Algoritmo 1 (C++ con OpenMP):** Paralelismo de datos sobre el bucle exterior de filas (`src/paralelo_b.cpp`).
2. **Algoritmo 2 (C++ con OpenMP):** Dividir y conquistar por cuadrantes usando tareas recursivas (`src/paralelo_c.cpp`).
3. **Algoritmo 3 (Python con Numba):** Paralelismo de datos con compilación JIT nativa `@njit(parallel=True)` y `prange` (`src/paralelo_d_numba.py`).

### 🎯 Estado Actual del Repositorio:
- **Computador 1 ya fue medido:** Los experimentos del primer computador (Apple A18 Pro, 6 cores heterogéneos) ya fueron completados con éxito y sus resultados están guardados en `resultados/computador_1/`.
- **Tu misión como Agente en esta máquina (Computador 2):**
  1. Detectar las características de hardware de este nuevo computador (núcleos físicos, núcleos lógicos, frecuencia de reloj, memoria RAM y arquitectura).
  2. Compilar los códigos fuente C++ con soporte OpenMP (`-O3 -march=native -fopenmp`).
  3. Ejecutar la suite completa de benchmarks para este equipo.
  4. Generar el reporte complementario `resultados/computador_2/INFORME_EXPERIMENTOS_COMPUTADOR_2.md` y su PDF.
  5. Contrastar los resultados con el Computador 1 para responder el **Apartado (g)** de la rúbrica.

---

## 2. Estructura Organizada de Archivos

```text
├── README.md                           # Guía general de uso
├── AGENTS.md                           # Este archivo (guía para agentes IA)
├── requirements.txt                    # Dependencias de Python
├── ejecutar.py                         # Acceso directo para correr la suite
├── .gitignore                          # Exclusiones de Git
│
├── docs/                               # Documentación y pauta oficial
│   └── tarea02.pdf                     # Enunciado oficial de la tarea
│
├── src/                                # Códigos fuente C++ y Python
│   ├── serial.cpp                      # Referencia serial base y partición por bloques
│   ├── paralelo_b.cpp                  # Algoritmo 1: OpenMP Paralelismo de datos (filas)
│   ├── paralelo_c.cpp                  # Algoritmo 2: OpenMP Dividir y Conquistar (tareas)
│   ├── paralelo_c_cuadrantes.cpp       # Variante cuadrantes
│   └── paralelo_d_numba.py             # Algoritmo 3: Python con Numba
│
├── scripts/                            # Herramientas de automatización y benchmarking
│   ├── ejecutar_experimentos.py        # Orquestador universal de experimentos
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

## 3. Instrucciones Paso a Paso para el Agente

Cuando un usuario te pida ejecutar o complementar los experimentos en su máquina (Computador 2), sigue estos pasos exactos:

### Paso 1: Instalar Dependencias de Python
```bash
pip install -r requirements.txt
```

### Paso 2: Verificar Compilador C++ y Soporte OpenMP
- **En Linux (Ubuntu / Debian):**
  ```bash
  sudo apt-get update && sudo apt-get install -y build-essential libomp-dev
  ```
- **En macOS (Homebrew):**
  ```bash
  brew install libomp
  ```
- **En Windows:** Asegurar `g++` con soporte OpenMP (vía MinGW-w64 / MSYS2).

### Paso 3: Ejecutar la Suite Completa en el Computador 2
Desde la raíz del repositorio:
```bash
python3 ejecutar.py --computador 2 --pdf
```
*Parámetros útiles:*
- `--computador 2`: Guarda los resultados en `resultados/computador_2/` e `INFORME_EXPERIMENTOS_COMPUTADOR_2.md`.
- `--reps 5`: Número de repeticiones estadísticas por punto de medición (por defecto: 5).
- `--pdf`: Compila automáticamente el informe generado a PDF estático vía Chrome headless.

### Paso 4: Validar la Consistencia de los Resultados
- **Verificación de Checksums:** Para $N = 2048$, el checksum de salida de la matriz $C$ **debe ser exactamente `172.331627`** en todos los algoritmos y con cualquier cantidad de hilos ($p=1, 2, 4, 8$). Si difiere, existe una condición de carrera.
- **Rango de Tiempos Esperados:**
  - $N = 1024$: ~50 a 300 ms.
  - $N = 2048$: ~500 a 2500 ms.
  - $N = 4096$: ~5 a 25 segundos.

---

## 4. Guía para el Análisis del Apartado (g) (Comparativa entre Equipos)

El informe generado en `resultados/computador_2/` incluirá automáticamente la tabla comparativa con el Computador 1. Al redactar el análisis para el usuario, enfócate en estos tres ejes fundamentales de HPC:

1. **Topología y Homogeneidad de Núcleos:**
   - *Computador 1 (Apple A18 Pro):* Arquitectura asimétrica big.LITTLE (2 núcleos Performance @ 4.05 GHz + 4 núcleos Efficiency @ 2.42 GHz). Con $p=1$ y $p=2$, corre a máxima frecuencia; con $p \ge 4$, los hilos en núcleos lentos rezagan la sincronización en las barreras OpenMP.
   - *Computador 2 (Tu máquina):* Si es x86 (Intel Core o AMD Ryzen), normalmente tiene núcleos homogéneos con Hyper-Threading/SMT. El escalamiento para $p=4$ y $p=8$ suele ser más lineal porque todos los núcleos ofrecen el mismo IPC y frecuencia.

2. **Caché y Ancho de Banda de Memoria:**
   - Apple Silicon utiliza memoria unificada LPDDR5X compartida y una caché L2 masiva (16 MB para solo 2 núcleos P).
   - En procesadores x86 de escritorio o laptop, la memoria DDR4/DDR5 de doble canal y cachés L3 compartidas de 16-32 MB influyen en el comportamiento para $N=4096$, donde las matrices ocupan >400 MB y saturan el bus de memoria.

3. **Cálculo de Eficiencia Débil Iso-granular:**
   - Dado que el costo es $\mathcal{O}(N^3)$, el trabajo se multiplica por 8 al duplicar $N$.
   - La condición de carga por hilo constante estricta es: $(N=1024, p=1) \to (N=2048, p=8)$.
   - $\mathcal{E}_{débil}^{iso} = \frac{T(1024, 1)}{T(2048, 8)} \times 100\%$.

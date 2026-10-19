# Resultados del Apple M5

Ejecución completa del 19/10/2026, de 16:23:35 a 16:33:55, hora de Santiago. Duración total: 621 segundos.

Se completaron 67 configuraciones con cinco repeticiones cada una: 335 mediciones. Se comprobaron 103 matrices completas pequeñas antes de medir; todas las repeticiones grandes pasaron el checksum independiente. No hubo errores en stderr.log.

## Equipo y método

Apple M5 (Mac17,2), 10 núcleos físicos y 10 lógicos: 4 de tipo Super y 6 Efficiency, según sysctl. Memoria de 16 GiB. macOS 26.6.2, Python 3.13.13. Compilador, flags, versiones de librerías y hashes están en metadata.json.

Cada punto usa un warm-up completo y luego cinco repeticiones. Los tiempos miden solo la multiplicación: inicialización, reserva, puesta a cero, checksum y JIT quedan fuera. Las desviaciones son muestrales. Speedup y eficiencia usan T(1) del mismo algoritmo, N y n.

El registro de inicio indica alimentación por batería al 33 %. El sistema no expone la frecuencia de CPU consultada: no se declara una frecuencia inventada. Estas condiciones deben quedar documentadas al comparar computadores.

Se ejecutó la variante de c por cuatro cuadrantes, coherente con el borrador. La variante alternativa de tareas recursivas no integra esta corrida. Los datos históricos del borrador no se mezclaron con estos resultados.

## Apartado a

| Bloque n | Promedio (ms) | Desviación (ms) |
| --- | ---: | ---: |
| 2048 | 1049,46 | 53,76 |
| 1024 | 1117,19 | 36,59 |
| 512 | 1124,22 | 59,10 |
| 256 | 1233,80 | 10,80 |
| 128 | 1248,50 | 1,72 |
| 64 | 1544,05 | 1,13 |
| 32 | 2178,03 | 3,72 |

## Apartados b c d y e

Comparación principal con N=2048; c usa n=64. Este valor se conservó como configuración común para el estudio de tamaños y la comparación con el protocolo preparado. No fue el bloque de menor tiempo medido en este equipo.

| Versión | p | Promedio (ms) | Desviación (ms) | Speedup | Eficiencia (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenMP datos | 1 | 970,49 | 7,36 | 1,00 | 100,00 |
| OpenMP datos | 2 | 595,88 | 11,42 | 1,63 | 81,43 |
| OpenMP datos | 4 | 370,70 | 6,89 | 2,62 | 65,45 |
| OpenMP datos | 8 | 288,55 | 6,20 | 3,36 | 42,04 |
| OpenMP cuadrantes | 1 | 1551,19 | 6,71 | 1,00 | 100,00 |
| OpenMP cuadrantes | 2 | 793,65 | 12,92 | 1,95 | 97,72 |
| OpenMP cuadrantes | 4 | 448,04 | 34,01 | 3,46 | 86,55 |
| OpenMP cuadrantes | 8 | 429,66 | 5,49 | 3,61 | 45,13 |
| Numba | 1 | 1029,31 | 38,78 | 1,00 | 100,00 |
| Numba | 2 | 551,54 | 11,75 | 1,87 | 93,31 |
| Numba | 4 | 373,56 | 4,77 | 2,76 | 68,89 |
| Numba | 8 | 305,86 | 10,26 | 3,37 | 42,07 |

A ocho threads, OpenMP datos y Numba tienen speedup cercano a 3,36, con tiempos de 288,55 ms y 305,86 ms. Cuadrantes obtiene 429,66 ms con n=64. Su mejoría entre cuatro y ocho threads es pequeña; su implementación solo reparte cuatro unidades de trabajo independientes.

## Barrido de bloques del apartado e

| p | Bloque con menor promedio | Promedio (ms) | Desviación (ms) |
| --- | ---: | ---: | ---: |
| 1 | 2048 | 1011,93 | 11,20 |
| 2 | 1024 | 549,92 | 9,86 |
| 4 | 512 | 337,41 | 6,51 |
| 8 | 512 | 334,43 | 6,48 |

Con cuatro threads, n=512 obtuvo 337,41 ms, frente a 448,04 ms con n=64: una reducción de tiempo de aproximadamente 24,7 %. Con ocho threads y n=512 se obtuvieron 334,43 ms, muy cerca del resultado con cuatro. La elección de n=64 del borrador no debe presentarse como óptima para este M5.

Los gráficos principales conservan n=64 para coincidir con las mediciones de f a los tres tamaños. El barrido permite discutir n=512 como alternativa más rápida en N=2048. Si se cambia el n común del informe, hay que mantener coherencia con las configuraciones usadas en f y en el otro computador.

## Apartado f

Eficiencia débil como T(N_base,1)/T(N_final,8). Al duplicar el lado de la matriz, el trabajo cúbico se multiplica por ocho. Se estudian dos pares independientes de carga N³/p constante.

| Versión | Par (N,p) | Eficiencia débil (%) |
| --- | --- | ---: |
| OpenMP datos | (1024,1) → (2048,8) | 42,94 |
| OpenMP datos | (2048,1) → (4096,8) | 28,02 |
| OpenMP cuadrantes | (1024,1) → (2048,8) | 42,27 |
| OpenMP cuadrantes | (2048,1) → (4096,8) | 40,11 |
| Numba | (1024,1) → (2048,8) | 39,49 |
| Numba | (2048,1) → (4096,8) | 32,27 |

Ninguna versión mantiene eficiencia débil cercana al 100 % en los pares medidos. El gráfico eficiencia_por_tamano.png es un análisis de eficiencia fuerte a cada N, complementario al gráfico de carga constante. No deben rotularse ambas métricas como equivalentes.

## Archivos y alcance

- mediciones.csv: las 335 repeticiones crudas.
- resumen.csv: las 67 configuraciones resumidas, con medias, desviaciones, T(1), S y E.
- escalabilidad_debil.csv: los seis cocientes de carga constante.
- graficos/: cuatro figuras revisadas visualmente.
- metadata.json, validacion.json y logs: entorno, estado y comprobaciones.
- fuentes/: copia de los códigos usados en esta corrida.

Estas mediciones cubren los experimentos preparados para este computador. Falta incorporar el segundo computador para completar g y la comparación entre equipos. El archivo Word original todavía no fue actualizado ni exportado a PDF.

Los resultados permiten describir las tendencias observadas, pero no identifican por sí solos un cuello de botella único de hardware. Caché, ancho de banda, núcleos heterogéneos, frecuencia y carga del sistema deben tratarse como posibles explicaciones cuando no hay mediciones directas que las distingan.

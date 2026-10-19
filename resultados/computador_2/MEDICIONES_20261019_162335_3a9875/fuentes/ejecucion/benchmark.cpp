// Adaptador de medición. Los kernels serial/c se incluyen sin modificar.
// ALGORITHM: 0 serial, 1 datos (nuevo apartado b), 2 tareas, 3 cuadrantes.
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <omp.h>

// Renombrar únicamente el main original permite reutilizar sus funciones.
#define main main_original_no_utilizado
#if ALGORITHM == 2
#include "paralelo_c.cpp"
#elif ALGORITHM == 3
#include "paralelo_c_cuadrantes.cpp"
#else
#include "serial.cpp"
#endif
#undef main

void ejecutar(const MatView& A, const MatView& B, const MatView& C, int n) {
#if ALGORITHM == 0
    multiply_rec(A, B, C, n);
#elif ALGORITHM == 1
    // Apartado b: cada hilo posee filas completas de C; k se acumula en orden.
    // Mismo orden ikj y precisión que el código del profesor.
    const int N = C.size;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            const double a = A.at(i, k);
            for (int j = 0; j < N; ++j)
                C.at(i, j) += a * B.at(k, j);
        }
#elif ALGORITHM == 2
    #pragma omp parallel
    {
        #pragma omp single
        multiply_rec_parallel(A, B, C, n, 0, 2);
    }
#else
    multiply_rec_parallel(A, B, C, n);
#endif
}

int main(int argc, char** argv) {
    try {
        if (argc < 5) throw std::runtime_error("Uso: binario N n p repeticiones [matriz.txt]");
        const int N = std::stoi(argv[1]), n = std::stoi(argv[2]);
        const int p = std::stoi(argv[3]), reps = std::stoi(argv[4]);
        if (N < 1 || N > 4096 || n < 1 || n > N || p < 1 || p > 8 || reps < 1 ||
            (N & (N-1)) || (n & (n-1)))
            throw std::runtime_error("Parámetros fuera de rango o tamaños no potencias de dos");
#if ALGORITHM == 0
        if (p != 1) throw std::runtime_error("Serial requiere p=1");
#endif
        omp_set_dynamic(0);
        omp_set_num_threads(p);
        int efectivos = 1;
#if ALGORITHM != 0
        #pragma omp parallel
        {
            #pragma omp single
            efectivos = omp_get_num_threads();
        }
#endif
        if (efectivos != p) throw std::runtime_error("El runtime no creó los threads solicitados");
        const size_t total = size_t(N) * N;
        std::vector<double> A(total), B(total), C(total);
        init_matrix(A, N, 1);
        init_matrix(B, N, 2);
        MatView Av{A.data(), N, N}, Bv{B.data(), N, N}, Cv{C.data(), N, N};
        // Una ejecución completa de calentamiento por configuración, sin medir.
        ejecutar(Av, Bv, Cv, n);
        for (int r = 1; r <= reps; ++r) {
            // Reservar, inicializar y poner a cero quedan fuera del cronómetro.
            std::fill(C.begin(), C.end(), 0.0);
            const auto t0 = std::chrono::steady_clock::now();
            ejecutar(Av, Bv, Cv, n);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
            double checksum = 0;
            for (double x : C) {
                if (!std::isfinite(x)) throw std::runtime_error("Resultado no finito");
                checksum += x;
            }
            std::cout << std::setprecision(17)
                << "{\"rep\":" << r << ",\"ms\":" << ms
                << ",\"checksum\":" << checksum << ",\"threads\":" << efectivos << "}" << std::endl;
        }
        if (argc > 5) {
            std::ofstream out(argv[5]);
            if (!out) throw std::runtime_error("No se pudo abrir el archivo de matriz");
            out << std::setprecision(17);
            for (double x : C) out << x << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}

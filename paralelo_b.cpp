// Tarea 2 IIC3533 - Apartado (b): Algoritmo 1 (Paralelismo de datos con OpenMP)
// Uso: ./paralelo_b [N p]
// Ejemplo: ./paralelo_b 2048 4

#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <omp.h>

// Valor determinista en [-1, 1] segun la posicion (identico a serial.cpp)
double det_value(int i, int j, double offset) {
    return std::sin(i * 0.1 + j * 0.3 + offset);
}

void init_matrix(std::vector<double>& M, int N, double offset) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            M[i * N + j] = det_value(i, j, offset);
        }
    }
}

// Algoritmo 1: Paralelismo de datos sobre el bucle exterior (i) con OpenMP
void matmul_omp(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double a = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 2048;
    int p = (argc > 2) ? std::atoi(argv[2]) : 4;

    if (N <= 0 || p <= 0) {
        std::fprintf(stderr, "Uso: %s [N p], con N > 0 y p >= 1\n", argv[0]);
        return 1;
    }

    std::vector<double> A(N * N), B(N * N), C(N * N, 0.0);
    init_matrix(A, N, 1);
    init_matrix(B, N, 2);

    omp_set_dynamic(0);
    omp_set_num_threads(p);

    auto t0 = std::chrono::high_resolution_clock::now();
    matmul_omp(A, B, C, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double checksum = 0.0;
    for (double v : C) {
        checksum += v;
    }

    std::printf("N=%d p=%d tiempo=%.2f ms checksum=%.6f\n", N, p, ms, checksum);
    return 0;
}

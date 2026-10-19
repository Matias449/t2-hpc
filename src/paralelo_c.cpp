// Tarea 2 IIC3533 - Apartado (c): dividir y conquistar con OpenMP.
// Uso: ./paralelo_c [N n p]
// Ejemplo: ./paralelo_c 2048 64 4
// N y n deben ser potencias de dos, con n <= N.

#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <omp.h>

struct MatView {
    double* data;
    int stride;
    int size;
    double& at(int i, int j) const { return data[i * stride + j]; }
    MatView sub(int r, int c, int s) const {
        return {data + r * stride + c, stride, s};
    }
};

double det_value(int i, int j, double offset) {
    return std::sin(i * 0.1 + j * 0.3 + offset);
}

void init_matrix(std::vector<double>& M, int N, double offset) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            M[i * N + j] = det_value(i, j, offset);
}

// Misma multiplicacion de hoja que en serial.cpp.
void multiply_leaf(const MatView& A, const MatView& B, const MatView& C) {
    int s = C.size;
    for (int i = 0; i < s; ++i)
        for (int k = 0; k < s; ++k) {
            double a = A.at(i, k);
            for (int j = 0; j < s; ++j)
                C.at(i, j) += a * B.at(k, j);
        }
}

// Version serial del profesor. Se utiliza bajo el limite de tareas para
// evitar crear un numero excesivo de tareas OpenMP.
void multiply_rec_serial(const MatView& A, const MatView& B,
                         const MatView& C, int n) {
    int s = C.size;
    if (s <= n) {
        multiply_leaf(A, B, C);
        return;
    }
    int h = s / 2;
    multiply_rec_serial(A.sub(0, 0, h), B.sub(0, 0, h), C.sub(0, 0, h), n);
    multiply_rec_serial(A.sub(0, h, h), B.sub(h, 0, h), C.sub(0, 0, h), n);
    multiply_rec_serial(A.sub(0, 0, h), B.sub(0, h, h), C.sub(0, h, h), n);
    multiply_rec_serial(A.sub(0, h, h), B.sub(h, h, h), C.sub(0, h, h), n);
    multiply_rec_serial(A.sub(h, 0, h), B.sub(0, 0, h), C.sub(h, 0, h), n);
    multiply_rec_serial(A.sub(h, h, h), B.sub(h, 0, h), C.sub(h, 0, h), n);
    multiply_rec_serial(A.sub(h, 0, h), B.sub(0, h, h), C.sub(h, h, h), n);
    multiply_rec_serial(A.sub(h, h, h), B.sub(h, h, h), C.sub(h, h, h), n);
}

// Crear tareas hasta una profundidad maxima evita exceso de overhead.
// Cada tarea es responsable de un CUADRANTE de C y realiza sus dos
// contribuciones EN ORDEN. Por ello no hay carreras de escritura sobre C.
void multiply_rec_parallel(const MatView& A, const MatView& B,
                           const MatView& C, int n,
                           int depth, int max_task_depth) {
    int s = C.size;
    if (s <= n || depth >= max_task_depth) {
        multiply_rec_serial(A, B, C, n);
        return;
    }
    int h = s / 2;
    #pragma omp taskgroup
    {
        #pragma omp task firstprivate(A, B, C, h, n, depth, max_task_depth)
        {
            multiply_rec_parallel(A.sub(0,0,h), B.sub(0,0,h), C.sub(0,0,h), n, depth+1, max_task_depth);
            multiply_rec_parallel(A.sub(0,h,h), B.sub(h,0,h), C.sub(0,0,h), n, depth+1, max_task_depth);
        }
        #pragma omp task firstprivate(A, B, C, h, n, depth, max_task_depth)
        {
            multiply_rec_parallel(A.sub(0,0,h), B.sub(0,h,h), C.sub(0,h,h), n, depth+1, max_task_depth);
            multiply_rec_parallel(A.sub(0,h,h), B.sub(h,h,h), C.sub(0,h,h), n, depth+1, max_task_depth);
        }
        #pragma omp task firstprivate(A, B, C, h, n, depth, max_task_depth)
        {
            multiply_rec_parallel(A.sub(h,0,h), B.sub(0,0,h), C.sub(h,0,h), n, depth+1, max_task_depth);
            multiply_rec_parallel(A.sub(h,h,h), B.sub(h,0,h), C.sub(h,0,h), n, depth+1, max_task_depth);
        }
        #pragma omp task firstprivate(A, B, C, h, n, depth, max_task_depth)
        {
            multiply_rec_parallel(A.sub(h,0,h), B.sub(0,h,h), C.sub(h,h,h), n, depth+1, max_task_depth);
            multiply_rec_parallel(A.sub(h,h,h), B.sub(h,h,h), C.sub(h,h,h), n, depth+1, max_task_depth);
        }
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 2048;
    int n = (argc > 2) ? std::atoi(argv[2]) : 64;
    int p = (argc > 3) ? std::atoi(argv[3]) : 4;
    int max_depth = (argc > 4) ? std::atoi(argv[4]) : 2;
    if (N <= 0 || n <= 0 || n > N || p <= 0 ||
        (N & (N-1)) != 0 || (n & (n-1)) != 0) {
        std::fprintf(stderr, "Uso: %s [N n p [max_depth]], con N y n potencias de 2, n <= N, p >= 1\n", argv[0]);
        return 1;
    }
    std::vector<double> A(N*N), B(N*N), C(N*N, 0.0);
    init_matrix(A, N, 1);
    init_matrix(B, N, 2);
    MatView Av{A.data(), N, N}, Bv{B.data(), N, N}, Cv{C.data(), N, N};
    omp_set_dynamic(0);
    omp_set_num_threads(p);

    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        #pragma omp single
        multiply_rec_parallel(Av, Bv, Cv, n, 0, max_depth);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double checksum = 0.0;
    for (double v : C) checksum += v;
    std::printf("N=%d n=%d p=%d tiempo=%.2f ms checksum=%.6f\n", N, n, p, ms, checksum);
}

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include "Eigen/Dense"

using namespace std;
using hr_clock = chrono::high_resolution_clock;
using seconds = chrono::duration<double>;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace Eigen;

inline size_t idx(size_t i, size_t j, size_t N) { return i * N + j; }

//построение а и ф
void build_A(vector<double>& A, size_t N)
{
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A[idx(i, j, N)] = 1.0 / (0.9 + 2.0 * static_cast<double>(i + 1)
                + static_cast<double>(j + 1));
}

void build_f(const vector<double>& A, vector<double>& f, size_t N)
{
    for (size_t i = 0; i < N; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < N; ++j) s += A[idx(i, j, N)];
        f[i] = s;                 // x* = (1 … 1)
    }
}

//LU
bool lu_decompose(vector<double>& A, vector<size_t>& piv, size_t N)
{
    piv.resize(N);
    for (size_t k = 0; k < N; ++k) {
        size_t max_row = k; double max_val = fabs(A[idx(k, k, N)]);
        for (size_t i = k + 1; i < N; ++i) {
            double v = fabs(A[idx(i, k, N)]);
            if (v > max_val) { max_val = v; max_row = i; }
        }
        if (max_val == 0.0) return false;
        piv[k] = max_row;
        if (max_row != k)
            for (size_t j = 0; j < N; ++j)
                swap(A[idx(k, j, N)], A[idx(max_row, j, N)]);
        for (size_t i = k + 1; i < N; ++i) {
            A[idx(i, k, N)] /= A[idx(k, k, N)];
            double lik = A[idx(i, k, N)];
            for (size_t j = k + 1; j < N; ++j)
                A[idx(i, j, N)] -= lik * A[idx(k, j, N)];
        }
    }
    return true;
}

void lu_solve(const vector<double>& LU, const vector<size_t>& piv,
    vector<double>& x, const vector<double>& b, size_t N)
{
    x = b;
    for (size_t k = 0; k < N; ++k)
        if (piv[k] != k) swap(x[k], x[piv[k]]);
    // Ly = Pb
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < i; ++j) x[i] -= LU[idx(i, j, N)] * x[j];
    // Ux = y
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
        for (size_t j = i + 1; j < N; ++j) x[i] -= LU[idx(i, j, N)] * x[j];
        x[i] /= LU[idx(i, i, N)];
    }
}

//QR
inline void apply_givens(double& a, double& b, double c, double s)
{
    double t = c * a - s * b;
    b = s * a + c * b;
    a = t;
}

bool qr_givens_solve(const vector<double>& A_in, const vector<double>& f_in,
    vector<double>& x, size_t N)
{
    vector<double> R = A_in;
    vector<double> b = f_in;
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = j + 1; i < N; ++i) {
            double a = R[idx(j, j, N)], bij = R[idx(i, j, N)];
            if (fabs(bij) < 1e-13) continue;
            double r = hypot(a, bij);
            double c = a / r, s = -bij / r;
            for (size_t k = j; k < N; ++k) {
                double Rjk = R[idx(j, k, N)], Rik = R[idx(i, k, N)];
                apply_givens(Rjk, Rik, c, s);
                R[idx(j, k, N)] = Rjk;
                R[idx(i, k, N)] = Rik;
            }
            apply_givens(b[j], b[i], c, s);
        }
    }
    for (size_t i = 0; i < N; ++i)
        if (fabs(R[idx(i, i, N)]) < 1e-15) return false;
    x.assign(N, 0.0);
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
        double s = b[i];
        for (size_t j = i + 1; j < N; ++j) s -= R[idx(i, j, N)] * x[j];
        x[i] = s / R[idx(i, i, N)];
    }
    return true;
}

//svd
constexpr double EPS = 1e-20;

double find_error(const VectorXd& x, const VectorXd& x_star) {
    return (x - x_star).norm() / x_star.norm();
}

void abs_singular_values(MatrixXd& Sigma, MatrixXd& U)
{
    int min_size = std::min(Sigma.rows(), Sigma.cols());
    for (int i = 0; i < min_size; ++i)
        if (Sigma(i, i) < 0) {
            Sigma(i, i) = -Sigma(i, i);
            U.col(i) = -U.col(i);
        }
}

void sort_singular_values(MatrixXd& Sigma, MatrixXd& U, MatrixXd& V)
{
    int min_size = std::min(Sigma.rows(), Sigma.cols());
    for (int I = 0; I < min_size; ++I) {
        double max_elem = Sigma(I, I);
        int idx_max = I;
        for (int i = I + 1; i < min_size; ++i)
            if (Sigma(i, i) > max_elem) { max_elem = Sigma(i, i); idx_max = i; }
        if (idx_max != I) {
            std::swap(Sigma(I, I), Sigma(idx_max, idx_max));
            U.col(I).swap(U.col(idx_max));
            V.col(I).swap(V.col(idx_max));
        }
    }
}

//house column
void column_transformation(MatrixXd& A, MatrixXd& U, int i, int j)
{
    VectorXd p(A.rows());
    double s = 0, beta, mu;
    for (int I = j; I < A.rows(); ++I) s += A(I, i) * A(I, i);
    if (std::sqrt(std::abs(s - A(j, i) * A(j, i))) > EPS) {
        beta = (A(j, i) < 0 ? std::sqrt(s) : -std::sqrt(s));
        mu = 1.0 / beta / (beta - A(j, i));
        p.setZero();
        for (int I = j; I < A.rows(); ++I) p(I) = A(I, i);
        p(j) -= beta;
        for (int m = 0; m < A.cols(); ++m) {
            s = 0;
            for (int n = j; n < A.rows(); ++n) s += A(n, m) * p(n);
            s *= mu;
            for (int n = j; n < A.rows(); ++n) A(n, m) -= s * p(n);
        }
        for (int m = 0; m < A.rows(); ++m) {
            s = 0;
            for (int n = j; n < A.rows(); ++n) s += U(m, n) * p(n);
            s *= mu;
            for (int n = j; n < A.rows(); ++n) U(m, n) -= s * p(n);
        }
    }
}

//house row
void row_transformation(MatrixXd& A, MatrixXd& V, int i, int j)
{
    VectorXd p(A.cols());
    double s = 0, beta, mu;
    for (int I = j; I < A.cols(); ++I) s += A(i, I) * A(i, I);
    if (std::sqrt(std::abs(s - A(i, j) * A(i, j))) > EPS) {
        beta = (A(i, j) < 0 ? std::sqrt(s) : -std::sqrt(s));
        mu = 1.0 / beta / (beta - A(i, j));
        p.setZero();
        for (int I = j; I < A.cols(); ++I) p(I) = A(i, I);
        p(j) -= beta;
        for (int m = 0; m < A.rows(); ++m) {
            s = 0;
            for (int n = j; n < A.cols(); ++n) s += A(m, n) * p(n);
            s *= mu;
            for (int n = j; n < A.cols(); ++n) A(m, n) -= s * p(n);
        }
        for (int m = 0; m < A.cols(); ++m) {
            s = 0;
            for (int n = j; n < A.cols(); ++n) s += V(m, n) * p(n);
            s *= mu;
            for (int n = j; n < A.cols(); ++n) V(m, n) -= s * p(n);
        }
    }
}

//под диагональ
void delete_elem_down_triangle(MatrixXd& A, MatrixXd& U, int I, int J)
{
    if (std::abs(A(I, J)) <= EPS) { A(I, J) = 0; return; }
    double r = std::sqrt(A(I, J) * A(I, J) + A(J, J) * A(J, J));
    double c = A(J, J) / r, s = A(I, J) / r;
    for (int k = 0; k < A.cols(); ++k) {
        double h1 = c * A(J, k) + s * A(I, k);
        double h2 = c * A(I, k) - s * A(J, k);
        A(J, k) = h1; A(I, k) = h2;
    }
    for (int k = 0; k < U.rows(); ++k) {
        double h1 = c * U(k, J) + s * U(k, I);
        double h2 = c * U(k, I) - s * U(k, J);
        U(k, J) = h1; U(k, I) = h2;
    }
    A(I, J) = 0;
}

//над диагональ
void delete_elem_up_triangle(MatrixXd& A, MatrixXd& V, int I, int J)
{
    if (std::abs(A(I, J)) <= EPS) { A(I, J) = 0; return; }
    double r = std::sqrt(A(I, J) * A(I, J) + A(I, I) * A(I, I));
    double c = A(I, I) / r, s = -A(I, J) / r;
    for (int k = 0; k < A.rows(); ++k) {
        double h1 = c * A(k, I) - s * A(k, J);
        double h2 = c * A(k, J) + s * A(k, I);
        A(k, I) = h1; A(k, J) = h2;
    }
    for (int k = 0; k < V.rows(); ++k) {
        double h1 = c * V(k, I) - s * V(k, J);
        double h2 = c * V(k, J) + s * V(k, I);
        V(k, I) = h1; V(k, J) = h2;
    }
}


void start_svd(const MatrixXd& A_in, MatrixXd& U, MatrixXd& Sigma, MatrixXd& V)
{
    Sigma = A_in;
    U = MatrixXd::Identity(A_in.rows(), A_in.rows());
    V = MatrixXd::Identity(A_in.cols(), A_in.cols());

    int min_size = std::min(Sigma.rows(), Sigma.cols());
    int up_size = min_size - 1;
    int down_size = min_size - 1;

    //бидиагонализация
    for (int i = 0; i < min_size - 1; ++i) {
        column_transformation(Sigma, U, i, i);
        row_transformation(Sigma, V, i, i + 1);
    }
    if (Sigma.rows() > Sigma.cols()) {
        column_transformation(Sigma, U, Sigma.cols() - 1, Sigma.cols() - 1);
        ++down_size;
    }
    if (Sigma.rows() < Sigma.cols()) {
        row_transformation(Sigma, V, Sigma.rows() - 1, Sigma.rows());
        ++up_size;
    }

    //итерации сведения к диагонали
    VectorXd up = VectorXd::Zero(up_size);
    VectorXd down = VectorXd::Zero(down_size);
    int cnt_up;
    do {
        cnt_up = 0;
        for (int i = 0; i < up_size; ++i) {
            if (std::abs(up[i] - Sigma(i, i + 1)) > EPS) {
                up[i] = Sigma(i, i + 1);
                delete_elem_up_triangle(Sigma, V, i, i + 1);
            }
            else ++cnt_up;
        }
        for (int i = 0; i < down_size; ++i) {
            if (std::abs(down[i] - Sigma(i + 1, i)) > EPS) {
                down[i] = Sigma(i + 1, i);
                delete_elem_down_triangle(Sigma, U, i + 1, i);
            }
        }
    } while (cnt_up != up_size);

    abs_singular_values(Sigma, U);
    sort_singular_values(Sigma, U, V);
}

int mrank(const MatrixXd& Sigma) { return Sigma.rows(); }
double cond(const MatrixXd& Sigma)
{
    int m = mrank(Sigma);
    return Sigma(0, 0) / Sigma(m - 1, m - 1);
}

struct Stat { double avg_time{ 0.0 }; double delta{ 0.0 }; };

Stat bench_LU(const vector<double>& A, const vector<double>& f, size_t N, int reps)
{
    const double norm_x_star = std::sqrt(static_cast<double>(N));
    double t_sum = 0.0, delta = 0.0;
    for (int r = 0; r < reps; ++r) {
        auto M = A; vector<size_t> piv; vector<double> x;
        auto t0 = hr_clock::now();
        lu_decompose(M, piv, N);
        lu_solve(M, piv, x, f, N);
        auto t1 = hr_clock::now();
        t_sum += seconds(t1 - t0).count();
        if (r == 0) {
            double diff2 = 0.0; for (double xi : x) diff2 += (xi - 1.0) * (xi - 1.0);
            delta = std::sqrt(diff2) / norm_x_star;
        }
    }
    return { t_sum / reps, delta };
}

Stat bench_QR(const vector<double>& A, const vector<double>& f, size_t N, int reps)
{
    const double norm_x_star = std::sqrt(static_cast<double>(N));
    double t_sum = 0.0, delta = 0.0;
    for (int r = 0; r < reps; ++r) {
        vector<double> x;
        auto t0 = hr_clock::now();
        qr_givens_solve(A, f, x, N);
        auto t1 = hr_clock::now();
        t_sum += seconds(t1 - t0).count();
        if (r == 0) {
            double diff2 = 0.0; for (double xi : x) diff2 += (xi - 1.0) * (xi - 1.0);
            delta = std::sqrt(diff2) / norm_x_star;
        }
    }
    return { t_sum / reps, delta };
}

Stat bench_SVD(const vector<double>& A_vec,
    const vector<double>& f_vec,
    size_t N, int reps)
{
    const double norm_x_star = sqrt(static_cast<double>(N));
    double t_sum = 0.0, delta = 0.0;
    VectorXd x_star = VectorXd::Ones(N);

    for (int r = 0; r < reps; ++r)
    {
        MatrixXd A(N, N); VectorXd f(N);
        for (size_t i = 0; i < N; ++i) {
            f(i) = f_vec[i];
            for (size_t j = 0; j < N; ++j) A(i, j) = A_vec[idx(i, j, N)];
        }

        auto t0 = hr_clock::now();
        Eigen::JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
        VectorXd sigma = svd.singularValues();

        double tol = 1e-12 * sigma(0);               // порог
        VectorXd sigmaInv = sigma.unaryExpr([&](double s) {
            return (fabs(s) > tol) ? 1.0 / s : 0.0;
            });

        VectorXd x = svd.matrixV() * sigmaInv.asDiagonal()
            * svd.matrixU().transpose() * f;
        auto t1 = hr_clock::now();
        t_sum += seconds(t1 - t0).count();

        if (r == 0) delta = (x - x_star).norm() / norm_x_star;
    }
    return { t_sum / reps, delta };
}

//запуск для н + вывод
void run_case(size_t N, int reps)
{
    vector<double> A(N * N), f(N);
    build_A(A, N);
    build_f(A, f, N);

    auto lu = bench_LU(A, f, N, reps);
    auto qr = bench_QR(A, f, N, reps);
    auto svd = bench_SVD(A, f, N, reps);

    MatrixXd A_eig(N, N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A_eig(i, j) = A[idx(i, j, N)];
    MatrixXd U, Sigma, V;
    start_svd(A_eig, U, Sigma, V);
    double condA = cond(Sigma);

    cout << "N = " << N << "\n";
    cout << "\t\t" << "Метод" << "\t\t" << "time, s" << "\t\t\t" << "б\n";
    cout << "\t\t" << "LU" << "\t\t" << fixed << setprecision(8) << lu.avg_time << scientific << "\t\t" << lu.delta << "\n";
    cout << "\t\t" << "QR" << "\t\t" << fixed << setprecision(8) << qr.avg_time << scientific << "\t\t" << qr.delta << "\n";
    cout << "\t\t" << "SVD" << "\t\t" << fixed << setprecision(8) << svd.avg_time << scientific << "\t\t" << svd.delta << "\n";
    cout << "cond(A) = " << scientific << condA << endl << endl << endl;
}

int main()
{
    setlocale(LC_ALL, "rus");
    ios::sync_with_stdio(false);
    const int REPS = 100;
    for (size_t N : { 5, 10, 20})
        run_case(N, REPS);

    return 0;
}

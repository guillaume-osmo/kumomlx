// Check the Eigen implementation against values computed in exact integer arithmetic.
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>
std::vector<double> characteristicPolynomial(const std::vector<std::vector<double>> &);
std::pair<double, double> ipcTerms(const std::vector<std::vector<double>> &);

static std::vector<std::vector<double>> chain(int n) {   // path graph = linear alkane
  std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
  for (int i = 0; i + 1 < n; ++i) { A[i][i + 1] = 1.0; A[i + 1][i] = 1.0; }
  return A;
}
int main() {
  std::printf("%5s %16s %16s %14s\n", "n", "last coeff", "AvgIpc", "log2(Ipc)");
  for (int n : {20, 60, 100, 120, 160, 200, 250}) {
    const auto A = chain(n);
    const auto c = characteristicPolynomial(A);
    const auto t = ipcTerms(A);
    std::printf("%5d %16.6f %16.6f %14.4f\n", n, c.back(), t.first, t.second);
  }
  return 0;
}

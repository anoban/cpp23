#include <vector>

template<typename T> [[nodiscard]] double sum(const std::vector<T>& collection) noexcept {
    double sum {};
    for (const auto& e : collection) sum += e;
    return sum;
}

[[nodiscard]] double sum(const std::vector<int>& collection) noexcept {
    double sum {};
    for (const auto& e : collection) sum += e;
    return sum;
}

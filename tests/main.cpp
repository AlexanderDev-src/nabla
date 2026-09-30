// Entry point for nabla_tests. The tests themselves live in the other
// tests/*.cpp files and register themselves through TEST(name).
#include "harness.hpp"
#include <cstdio>

int main() {
    // an empty registry would otherwise look exactly like a clean run
    if (nabla_test::registry().empty()) {
        std::printf("no tests registered\n");
        return 1;
    }
    return nabla_test::run_all() == 0 ? 0 : 1;
}

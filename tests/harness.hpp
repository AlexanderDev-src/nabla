#pragma once
#include <cmath>
#include <cstdio>
#include <exception>
#include <iostream>
#include <vector>

namespace nabla_test {

struct TestCase {
    const char *name;
    void (*fn)();
};

inline std::vector<TestCase> &registry() {
    static std::vector<TestCase> tests;
    return tests;
}

// Failed CHECKs in the test that is running right now.
inline int &current_failures() {
    static int count = 0;
    return count;
}

struct Registrar {
    Registrar(const char *name, void (*fn)()) {
        registry().push_back({name, fn});
    }
};

inline int run_all() {
    int failed_count = 0;
    const int total_tests = static_cast<int>(registry().size());

    for (const auto &t : registry()) {
        current_failures() = 0;
        bool has_thrown = false;

        try {
            t.fn();
        } catch (const std::exception &e) {
            std::cout << "  threw: " << e.what() << "\n";
            has_thrown = true;

        } catch (...) {
            has_thrown = true;
        }

        if (has_thrown || current_failures() > 0) {
            std::cout << "[FAIL] " << t.name << "\n";
            ++failed_count;
        }
    }
    const int passed_count = total_tests - failed_count;
    std::cout << passed_count << "/" << total_tests << " tests passed\n";

    return failed_count;
}

} // namespace nabla_test

#define TEST(name)                                                             \
    static void test_##name();                                                 \
    static const nabla_test::Registrar registrar_##name(#name, test_##name);   \
    static void test_##name()

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
            ++nabla_test::current_failures();                                  \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b) CHECK(std::fabs((a) - (b)) < 1e-5f)

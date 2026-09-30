#pragma once
// A small test harness: TEST(name) { ... } registers itself, run_all()
// runs every registered test and reports which ones failed.
#include <bits/stdc++.h>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace nabla_test {

struct TestCase {
    const char *name;
    void (*fn)();
};

// Every TEST lands here. A function-local static is built the first time
// the function runs, so the list exists before any Registrar needs it, no
// matter in which order the globals of different .cpp files are set up.
inline std::vector<TestCase> &registry() {
    static std::vector<TestCase> tests;
    return tests;
}

// Failed CHECKs in the test that is running right now.
inline int &current_failures() {
    static int count = 0;
    return count;
}

// TEST(name) creates one global Registrar. Globals are constructed before
// main() starts, so each test adds itself to the registry without anyone
// calling it by hand.
struct Registrar {
    Registrar(const char *name, void (*fn)()) {
        registry().push_back({name, fn});
    }
};

// Runs every registered test in order and returns how many tests failed.
// A test fails when any CHECK inside it fails, or when it throws.
inline int run_all() {
    // TODO: (2) for each test in registry():
    //           - set current_failures() back to 0
    //           - call the test's fn inside try/catch; an exception counts
    //             as a failure and must not stop the tests after it
    //           - if the test failed, print "[FAIL] <name>" and count it
    //       after the loop, print a summary such as "29/31 tests passed"
    //       and return the number of failed tests
    int failed_count = 0;
    const int total_tests = static_cast<int>(registry().size());

    for (const auto &t : registry()) {
        current_failures() = 0;
        bool has_thrown = false;

        try {
            t.fn();
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

// #name turns the identifier into a string literal, and test_##name pastes
// it into a new identifier, so TEST(zeros) declares test_zeros(), registers
// it under the name "zeros", and then opens the body of test_zeros().
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

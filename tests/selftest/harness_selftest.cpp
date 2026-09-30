// Checks the harness in tests/harness.hpp against tests whose outcome is
// known in advance: one passes, one fails two CHECKs, one throws, and one
// runs after the throw. The FAIL lines these tests print are expected.
#include "harness.hpp"
#include <cstdio>
#include <stdexcept>

static bool ran_after_throw = false;

TEST(passes) { CHECK(1 + 1 == 2); }

TEST(expected_fail_two_checks) {
    CHECK(1 + 1 == 3);
    CHECK(2 + 2 == 5);
}

TEST(expected_fail_throw) { throw std::runtime_error("thrown on purpose"); }

TEST(runs_after_throw) { ran_after_throw = true; }

int main() {
    int bad = 0;
    auto expect = [&bad](bool ok, const char *what) {
        if (!ok) {
            std::printf("SELFTEST FAIL: %s\n", what);
            ++bad;
        }
    };

    expect(nabla_test::registry().size() == 4,
           "all 4 tests are in the registry");

    std::printf("--- run_all output (2 failing tests are expected) ---\n");
    const int failed = nabla_test::run_all();
    std::printf("--- end of run_all output ---\n");

    expect(failed == 2, "run_all returns 2: it counts failing tests, not "
                        "failing CHECKs, and resets the count between tests");
    expect(ran_after_throw, "the test after a throwing test still runs");

    std::printf(bad ? "\nharness selftest FAILED\n" : "\nharness selftest ok\n");
    return bad;
}

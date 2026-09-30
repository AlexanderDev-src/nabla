cmake_minimum_required(VERSION 3.15)
project(nabla LANGUAGES CXX)

# 1. Global settings first
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# 2. Sources (every .cpp under src/ is picked up automatically)
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

# 3. Create the executable first ...
add_executable(nabla ${SOURCE_FILES})

# 4. ... then attach include paths and warnings to it
target_include_directories(nabla PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(nabla PRIVATE -Wall -Wextra -Wpedantic)

# 5. Test executable (separate target, shares the same headers). Only the
#    files directly in tests/ go in: tests/selftest/ has its own main().
file(GLOB TEST_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tests/*.cpp")
add_executable(nabla_tests ${TEST_FILES})
target_include_directories(nabla_tests PUBLIC ${CMAKE_SOURCE_DIR}/include
                                              ${CMAKE_SOURCE_DIR}/tests)
target_compile_options(nabla_tests PRIVATE -Wall -Wextra -Wpedantic)

# 6. Self-test for the harness in tests/harness.hpp
add_executable(harness_selftest ${CMAKE_SOURCE_DIR}/tests/selftest/harness_selftest.cpp)
target_include_directories(harness_selftest PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_compile_options(harness_selftest PRIVATE -Wall -Wextra -Wpedantic)

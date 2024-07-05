include(ExternalProject)

ExternalProject_Add(benchmark-external
    PREFIX          ${CMAKE_CURRENT_BINARY_DIR}
    DOWNLOAD_DIR    ${MyBench_SOURCE_DIR}/contrib/benchmark-cmake/download
    URL             https://github.com/google/benchmark/archive/refs/tags/v1.8.4.zip
    URL_HASH        MD5=e2cb17901062de42e0fc41844e4b9ac0
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DBENCHMARK_ENABLE_TESTING=OFF
        -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
        -DBENCHMARK_ENABLE_WERROR=OFF
        -DBENCHMARK_INSTALL_DOCS=OFF
    BUILD_BYPRODUCTS        <INSTALL_DIR>/lib/${CMAKE_FIND_LIBRARY_PREFIXES}benchmark.a  # Workaround for Ninja
    USES_TERMINAL_DOWNLOAD  TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
    USES_TERMINAL_INSTALL   TRUE
    EXCLUDE_FROM_ALL        TRUE
)

ExternalProject_Get_Property(benchmark-external INSTALL_DIR)

add_library(contrib::benchmark STATIC IMPORTED GLOBAL)
set_target_properties(contrib::benchmark PROPERTIES
    IMPORTED_LOCATION       ${INSTALL_DIR}/lib/${CMAKE_FIND_LIBRARY_PREFIXES}benchmark.a
)
add_dependencies(contrib::benchmark benchmark-external)

file(MAKE_DIRECTORY ${INSTALL_DIR}/include)
target_include_directories(contrib::benchmark SYSTEM INTERFACE
    ${INSTALL_DIR}/include
)

set(HIGHFIVE_PROJECT_DIR "${MyBench_SOURCE_DIR}/contrib/highfive")
set(HIGHFIVE_SOURCE_DIR "${HIGHFIVE_PROJECT_DIR}/include")

if (NOT EXISTS "${HIGHFIVE_SOURCE_DIR}/highfive/highfive.hpp")
    message (FATAL_ERROR "submodule contrib/highfive not found")
endif()

add_library(_highfive INTERFACE)

target_include_directories(_highfive SYSTEM INTERFACE
    ${HIGHFIVE_SOURCE_DIR}
)

target_link_libraries(_highfive INTERFACE
    contrib::hdf5
)

add_library(contrib::highfive ALIAS _highfive)

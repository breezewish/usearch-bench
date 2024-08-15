#pragma once

// Note: Be careful that usearch also includes simsimd with a customized config.
// Don't include simsimd and usearch at the same time. Otherwise, the effective
// config depends on the include order.
#define SIMSIMD_NATIVE_F16 0
#define SIMSIMD_NATIVE_BF16 0
#define SIMSIMD_DYNAMIC_DISPATCH 0

// Force enable all target features. We will do our own dynamic dispatch.
#define SIMSIMD_TARGET_NEON 1
#define SIMSIMD_TARGET_SVE 1
#define SIMSIMD_TARGET_HASWELL 1
#define SIMSIMD_TARGET_SKYLAKE 1
#define SIMSIMD_TARGET_ICE 1
#define SIMSIMD_TARGET_GENOA 1
#define SIMSIMD_TARGET_SAPPHIRE 1
#include <simsimd/simsimd.h>

// CUB include shim for the HIP build: route <cub/cub.cuh> to hipCUB. The
// compat header additionally aliases the cub:: namespace token to hipcub::.
// On the NVIDIA build this dir is not on the include path, so the real CUB is
// used unchanged.
#pragma once
#include <hipcub/hipcub.hpp>

#ifndef SOLVER_H
#define SOLVER_H


#include "Common.h"
#include "Device.h"


// Is number "n" power of "base"?
__forceinline__ __host__ __device__  bool h_d_checkPower(num_t n, base_t base)
{
    num_t i;
    for (i = base; i < n; i *= base);
    return i == n;
}

__host__ result_t host_solve_task(num_t n);
__host__ result_t device_solve_task
(
    const Device& dev,
    num_t n,
    const dim3& globalDim3,
    cudaError_t* const cudaStatus
);

#endif
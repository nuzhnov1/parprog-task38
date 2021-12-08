#ifndef COMMON_H
#define COMMON_H


#include <iostream>
#include <cstdint>
#include <utility>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


constexpr size_t WARP_SIZE          = 32;
constexpr size_t MAX_THREADS        = 1024;
constexpr size_t BLOCK_DIM_X_MAX    = (1 << 16) - 1;


using num_t     = unsigned long long int;
using base_t    = unsigned int;


struct result_t
{
    num_t num;
    num_t sum;
    base_t base;
};

inline bool operator==(const result_t& f_result, const result_t& s_result)
{
    return f_result.num == s_result.num &&
           f_result.sum == s_result.sum &&
           f_result.base == s_result.base;
}


// Is print message to stderr?
#define VALIDATE_CUDA VALIDATE_CUDA_PRINT  // yes

#define VALIDATE_CUDA_PRINT(status, errmsg, ret)                        \
do                                                                      \
{                                                                       \
    if ((status) != cudaSuccess)                                        \
    {                                                                   \
        std::cerr << (errmsg) << std::endl;                             \
        std::cerr << "Error: " << cudaGetErrorString(status) << "[" <<  \
            cudaGetErrorName(status) << "]" << std::endl;               \
                                                                        \
        return ret;                                                     \
    }                                                                   \
} while (0);

#define VALIDATE_CUDA_NO_PRINT(status, ret)                             \
do                                                                      \
{                                                                       \
    if ((status) != cudaSuccess)                                        \
        return ret;                                                     \
} while (0);


#endif  // COMMON_H

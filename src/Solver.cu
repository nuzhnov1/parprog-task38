#include "Solver.cuh"

#include <cmath>
#include <limits>


// Auxiliary functions on CPU and GPU
///////////////////////////////////////////////////////////////////////////////
__forceinline__ __host__ __device__ unsigned int _h_d_getNumberLen(num_t n)
{
    unsigned int result = 0;

    while (n > 0)
    {
        n /= 10;
        result++;
    }

    return result;
}
///////////////////////////////////////////////////////////////////////////////


// CPU implementation:
///////////////////////////////////////////////////////////////////////////////
__host__ num_t h_getSumCompactSubSequences(num_t n)
{
    num_t result = 0;  // sum of all compact subsequences
    unsigned int number_len = _h_d_getNumberLen(n);  // lenth of number "n"

    // Iterate over all possible lengths of compact subsequences of the
    // number "n": [1, 2, ... , length of number - 1]
    for (unsigned int seq_len = 1; seq_len < number_len; seq_len++)
    {
        // Iterate over all possible compact subsequences with current length
        for
        (
            unsigned int i = 0, end = number_len - (seq_len - 1);
            i < end;
            i++
        )
        {
            // Cut off the left digits
            num_t sub_seq = n % (num_t)(pow(10UL, (number_len - i)));
            // Cut off the right digits
            sub_seq = sub_seq / (num_t)(pow(10UL, (number_len - seq_len - i)));

            result += sub_seq;
        }
    }

    return result;
}

__host__ base_t _h_findFirstBase(num_t n)
{
    for (base_t base = 2, end = ceil(sqrt((double)n)); base < end; base++)
        if (h_d_checkPower(n, base))
            return base;
    
    return 0;
}

__host__ result_t h_solve_task(num_t n)
{
    for (n++; true; n++)
    {
        num_t sum = h_getSumCompactSubSequences(n);
        base_t base;

        if ((base = _h_findFirstBase(sum)) != 0)
            return {n, sum, base};
    }
}
////////////////////////////////////////////////////////////////////////////////


// GPU implementation:
///////////////////////////////////////////////////////////////////////////////
__device__ void _d_getSumCompactSubSequences(num_t n, num_t* result)
{
    // Length of number "n"
    const unsigned int number_len = _h_d_getNumberLen(n);
    // Required number of threads
    const unsigned int number_threads = (number_len + 2) * (number_len - 1) / 2;

    // Count of subsequences for each thread 
    const unsigned int seq_count = number_threads / blockDim.x;
    // Rest of subsequences
    const unsigned int seq_rest = number_threads % blockDim.x;

    // Thread id
    const unsigned int tid = threadIdx.x;

    // Threads with an id greater than the maximum leave
    // the execution of the function
    if (tid >= number_threads)
        return;

    unsigned int seq_id;   // Subsequence id
    unsigned int seq_len;  // Length of current subsequence
    unsigned int temp_len;

    // Get a subsubsequence id and length
    seq_id = tid * seq_count;
    seq_len = 1;
    temp_len = number_len;
    for (; seq_id >= temp_len; temp_len--, seq_len++)
        seq_id -= temp_len;
    seq_id = seq_id % temp_len;

    // A cycle over a group of subsequences
    for (unsigned int i = 0; i < seq_count; i++)
    {
        // Cut off the left digits
        num_t sub_seq = n % (num_t)(pow(10, (number_len - seq_id)));
        // Cut off the right digits
        sub_seq = sub_seq / (num_t)(pow(10, (number_len - seq_len - seq_id)));

        // Adding the resulting subsequence to the result
        atomicAdd(result, sub_seq);

        // Move to next subsequence
        seq_id++;
        if (seq_id == temp_len)
        {
            seq_id = 0;
            seq_len++;
            temp_len--;
        }
    }

    // A cycle over a rest subsequences
    if (tid < seq_rest)
    {
        // Get a subsequence id and length
        seq_id = seq_count * blockDim.x + tid;
        seq_len = 1;
        temp_len = number_len;
        for (; seq_id >= temp_len; temp_len--, seq_len++)
            seq_id -= temp_len;
        seq_id = seq_id % temp_len;

        // Cut off the left digits
        num_t sub_seq = n % (num_t)(pow(10, (number_len - seq_id)));
        // Cut off the right digits
        sub_seq = sub_seq / (num_t)(pow(10, (number_len - seq_len - seq_id)));

        // Adding the resulting subsequence to the result
        atomicAdd(result, sub_seq);
    }
}

__device__ void _d_findFirstBase(num_t n, base_t* result)
{
    // The nearest larger integer greater than the square root of n
    const unsigned int c_sqrt = (unsigned int)(ceil(sqrt((double)n)));
    // The size of a group of numbers for each thread
    const unsigned int num_count = (c_sqrt < 2) ? 0 : 
        (c_sqrt - 2) / blockDim.x;
    // The size of a rest numbers
    const unsigned int num_rest = (c_sqrt < 2) ? c_sqrt : 
        (c_sqrt - 2) % blockDim.x;
    // Thread id
    const unsigned int tid = threadIdx.x;

    // Processing all the numbers from the group
    base_t base = tid * num_count + 2;  // First base
    base_t end = base + num_count; 
    for (; base < end; base++)
    {
        if (h_d_checkPower(n, base))
            // Write the minumum base to result
            atomicMin(result, base);
    }

    // Processing all the rest numbers
    if (tid < num_rest)
    {
        base_t base = blockDim.x * num_count + 2 + tid;

        if (h_d_checkPower(n, base))
            // Write the minumum base to result
            atomicMin(result, base);
    }
}

__global__ void _d_solve_task(num_t n, result_t* g_results)
{
    extern __shared__ result_t s_results[];  // Solving results

    const unsigned int tidx = threadIdx.x;   // ID of thread id in local group
    const unsigned int tidy = threadIdx.y;   // ID of local group in block
    const unsigned int bid = blockIdx.x;     // ID of block

    // Value of "n" for current local group
    n = n + bid * blockDim.y + tidy + 1;

    // Initializing shared results for each local group
    if (tidx == 0)
    {
        s_results[tidy].num = n;
        s_results[tidy].sum = 0;
        s_results[tidy].base = std::numeric_limits<base_t>::max();
    }
    __syncthreads();
    
    _d_getSumCompactSubSequences(n, &s_results[tidy].sum);
    __syncthreads();
    _d_findFirstBase(s_results[tidy].sum, &s_results[tidy].base);

    // One thread(with id == 0) on each group checks result
    if (tidx == 0)
    {
        // If the sum is not a power of any number, write to num invalid value
        // - 0
        if (s_results[tidy].base == std::numeric_limits<base_t>::max())
            s_results[tidy].num = std::numeric_limits<num_t>::max();
    }
    // Other threads on current group leave the execution of this kernel
    else
        return;
    
    __syncthreads();

    // Finding the minimum value of "s_results" array and write it to zero item
    for (unsigned int step = 1; step < blockDim.y; step *= 2)
    {
        if (tidy % (2 * step) == 0)
        {
            if (s_results[tidy + step].num < s_results[tidy].num)
                s_results[tidy] = s_results[tidy + step];
        }
        else
            return;
    }

    // Write minimum value to global memory
    g_results[bid] = s_results[0];
}

__global__ void _d_minReduce(unsigned int size, result_t* g_results)
{
    extern __shared__ result_t s_results[];

    // Count of array items for each thread
    const unsigned int count = size / blockDim.x;
    // Scaled thread id
    const unsigned int stid = count * threadIdx.x;

    if (stid >= size)
        return;

    // Write data from global to shared memory
    for (unsigned int i = 0; i < count; i++)
        s_results[stid + i] = g_results[stid + i];
    __syncthreads();

    // Find the minimum value among the array items for each thread
    for (unsigned int i = 1; i < count; i++)
        if (s_results[stid + i].num < s_results[stid].num)
            s_results[stid] = s_results[stid + i];
    
    // Find the minimum value of "s_results" array and write it to zero item
    for (unsigned int step = 1; step < size; step *= 2)
    {
        if (stid % (2 * step) == 0)
        {
            if (s_results[stid + step].num < s_results[stid].num)
                s_results[stid] = s_results[stid + step];
        }
        else
            return;
    }

    // Write minimum value to first item in global memory
    g_results[0] = s_results[0];
}


#define WRITE_STATUS(ptr, error)    \
do                                  \
{                                   \
    if ((cudaStatus) != nullptr)    \
        *(ptr) = (error);           \
} while (0);

#define SAFE_RELEASE_CUDA_MEM(ptr) cudaFree(ptr); (ptr) = nullptr;

#define VALIDATE_AND_RELEASE(status, ptr)   \
do                                          \
{                                           \
    if ((status) != cudaSuccess)            \
    {                                       \
        SAFE_RELEASE_CUDA_MEM(ptr);         \
    }                                       \
} while (0);

__host__ result_t h_start_kernel
(
    num_t n,
    const dim3& localDim3, const dim3& globalDim3,
    cudaError_t* const cudaStatus
)
{
    cudaError_t cudaError;
    // Write to h_result invalid value
    result_t h_result = {std::numeric_limits<num_t>::max(), 0, 0};
    result_t* d_results;

    unsigned int blocks_count = globalDim3.x;

    cudaError = cudaMalloc(&d_results, sizeof(h_result) * blocks_count);
    WRITE_STATUS(cudaStatus, cudaError);
    VALIDATE_CUDA
    (
        cudaError,
        "Failed to allocate memory on device.",
        h_result
    );

    // Main loop
    for
    (
        ;
        // While h_result is invalid:
        h_result.num == std::numeric_limits<num_t>::max();
        // Move to next numbers group
        n += localDim3.y * blocks_count
    )
    {
        // Starting main kernel. Each CUDA block find the minimum value of "n"
        // in it's numbers group and writes result to global memory
        _d_solve_task
        <<<
            globalDim3, localDim3, sizeof(h_result) * localDim3.y
        >>>(n, d_results);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        WRITE_STATUS(cudaStatus, cudaError);
        VALIDATE_AND_RELEASE(cudaError, d_results);
        VALIDATE_CUDA
        (
            cudaError, "Failed to run '_d_solve_task' kernel.", h_result
        );

        // Starting minReduce kernel. One CUDA block reads all intermidiate
        // results from global memory, finds the minimum among them and write
        // it to first global memory item
        _d_minReduce
        <<<
            1, 1024, sizeof(h_result) * blocks_count
        >>>(blocks_count, d_results);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        WRITE_STATUS(cudaStatus, cudaError);
        VALIDATE_AND_RELEASE(cudaError, d_results);
        VALIDATE_CUDA
        (
            cudaError, "Failed to run '_d_minReduce' kernel.", h_result
        );

        // Copy to h_result zero item of d_results
        cudaError = cudaMemcpy
        (
            &h_result, d_results,
            sizeof(h_result),
            cudaMemcpyDeviceToHost
        );
        WRITE_STATUS(cudaStatus, cudaError);
        VALIDATE_AND_RELEASE(cudaError, d_results);
        VALIDATE_CUDA
        (
            cudaError,
            "Failed to copy memory from device to host.",
            h_result
        );
    }

    SAFE_RELEASE_CUDA_MEM(d_results);
    return h_result;
}
///////////////////////////////////////////////////////////////////////////////

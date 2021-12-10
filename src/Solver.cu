#include "Solver.cuh"

#include <cmath>
#include <limits>
#include <algorithm>


// Auxiliary functions on Host and Device
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
    unsigned int number_len = _h_d_getNumberLen(n);  // lenth of number n

    // Iterate over all possible lengths of compact subsequences of the
    // number n: [1, 2, ... , length of number - 1]
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

__host__ result_t host_solve_task(num_t n)
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
__device__ void _d_getSumCompactSubSequences
(
    num_t n, unsigned int tid, unsigned int threads_count,
    num_t* const result
)
{
    // Length of number n
    const unsigned int number_len = _h_d_getNumberLen(n);
    // Required number of threads
    const unsigned int required_threads = (number_len + 2) * 
                                          (number_len - 1) / 2;

    // Count of subsequences for each thread 
    const unsigned int seq_count = required_threads / threads_count;
    // Rest of subsequences
    const unsigned int seq_rest = required_threads % threads_count;

    // Threads with an id greater than the maximum leave
    // the execution of the function
    if (tid >= required_threads)
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
        seq_id = seq_count * threads_count + tid;
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

__device__ void _d_findFirstBase
(
    num_t n, unsigned int tid, unsigned int threads_count,
    base_t* const result
)
{
    // The nearest larger integer greater than the square root of n
    const unsigned int c_sqrt = (unsigned int)(ceil(sqrt((double)n)));
    // The size of a group of numbers for each thread
    const unsigned int num_count = (c_sqrt < 2) ? 0 : 
        (c_sqrt - 2) / threads_count;
    // The size of a rest numbers
    const unsigned int num_rest = (c_sqrt < 2) ? c_sqrt : 
        (c_sqrt - 2) % threads_count;

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
        base_t base = threads_count * num_count + 2 + tid;

        if (h_d_checkPower(n, base))
            // Write the minumum base to result
            atomicMin(result, base);
    }
}

__global__ void _d_solve_task(num_t n, result_t* const g_results)
{
    // WARNING: blockIdx.y must be equal to warp size!
    // So blockIdx.x = MAX_THREADS / (warp size) - count of warps in block
    
    extern __shared__ result_t s_results[];  // Solving intermidiate results

    const unsigned int bid = blockIdx.x;   // ID of block in grid
    const unsigned int wid = threadIdx.y;  // ID of warp in block
    const unsigned int tid = threadIdx.x;  // ID of thread in warp

    const unsigned int warp_count = blockDim.y;  // Also it's size of s_results
    const unsigned int warp_size  = blockDim.x;  // Count of threads in warp

    // Value of n for current warp
    n = n + bid * warp_count + wid + 1;

    // Initializing shared results for each warp
    if (tid == 0)
    {
        s_results[wid].num = n;
        s_results[wid].sum = 0;
        s_results[wid].base = std::numeric_limits<base_t>::max();
    }
    __syncwarp();
    
    _d_getSumCompactSubSequences(n, tid, warp_size, &s_results[wid].sum);
    __syncwarp();
    _d_findFirstBase(s_results[wid].sum, tid, warp_size, &s_results[wid].base);

    // One thread(with id == 0) on each warp checks result,
    // other threads leave execution of kernel 
    if (tid == 0)
    {
        // If the sum is not a power of any number, write to num invalid value
        // - maximum number
        if (s_results[wid].base == std::numeric_limits<base_t>::max())
            s_results[wid].num = std::numeric_limits<num_t>::max();
    }
    else
        return;
    
    __syncthreads();

    // Finding the minimum value of s_results array and write it to zero item
    for (unsigned int step = 1; step < warp_count; step *= 2)
    {
        if (wid % (2 * step) == 0)
        {
            if (s_results[wid + step].num < s_results[wid].num)
                s_results[wid] = s_results[wid + step];
        }

        __syncthreads();
    }

    // Write minimum value to global memory
    if (wid == 0)
        g_results[bid] = s_results[0];
}

__global__ void _d_minReduce
(
    size_t size, const result_t* input,
    result_t* const output
)
{
    extern __shared__ result_t s_results[];

    // Global id of thread in device grid:
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int bid = blockIdx.x;   // ID of block
    const unsigned int tid = threadIdx.x;  // ID of thread in block

    // All threads with gid greater then size - leave the execution of kernel
    if (gid >= size)
        return;

    // Write data from input global memory to shared one
    s_results[tid] = input[gid];
    __syncthreads();

    // Find the minimum value of s_results array and write it to zero item
    for (unsigned int step = 1; step < blockDim.x; step *= 2)
    {
        if ((tid % (2 * step)) == 0 && (tid + step < blockDim.x))
        {
            if (s_results[tid + step].num < s_results[tid].num)
                s_results[tid] = s_results[tid + step];
        }

        __syncthreads();
    }

    // Write minimum value to output global memory
    if (tid == 0)
        output[bid] = s_results[0];
}

__host__ result_t _h_minReduce
(
    const Device& dev,
    size_t size, result_t* d_array,
    cudaError_t* const cudaStatus
)
{
    // chunk_size must be not greater than max_threads
    // So maximum value of "size" is 2 ^ 20

    cudaError_t cudaError;
    result_t* d_interm_results;  // intermidiate results on device
    result_t* d_result;          // final result on device
    result_t h_result = {std::numeric_limits<num_t>::max(), 0, 0};

    unsigned int max_threads    = dev.getMaxThreadsDim()[0];
    size_t max_shared_mem_size  = dev.getSharedPerBlockMem();
    unsigned int threads_count  = max_threads;

    // If there is not enough shared memory for each block - 
    // reduce the number of threads
    if (max_threads * sizeof(result_t) >= max_shared_mem_size - 512)
        threads_count = (max_shared_mem_size - 512) / sizeof(result_t);
    
    unsigned int chunk_size = ceil((double)size / threads_count);

    if (chunk_size == 0)
        return h_result;

    // Allocating device memory to intermidiate results
    cudaError = cudaMalloc(&d_interm_results, sizeof(result_t) * chunk_size);
    WRITE_STATUS(cudaStatus, cudaError);
    VALIDATE_CUDA
    (
        cudaError,
        "Failed to allocate memory on device.",
        h_result
    );

    // Allocating device memory to final result
    cudaError = cudaMalloc(&d_result, sizeof(result_t));
    WRITE_STATUS(cudaStatus, cudaError);
    VALIDATE_AND_RELEASE(cudaError, d_interm_results);
    VALIDATE_CUDA
    (
        cudaError,
        "Failed to allocate memory on device.",
        h_result
    );

    // Starting '_d_minReduce' kernel to get intermidiate results
    threads_count = std::min((unsigned int)size, max_threads);
    _d_minReduce
    <<<
        chunk_size,
        threads_count,
        sizeof(result_t) * threads_count
    >>>(size, d_array, d_interm_results);
    cudaDeviceSynchronize();
    cudaError = cudaGetLastError();
    WRITE_STATUS(cudaStatus, cudaError);
    VALIDATE_AND_RELEASE(cudaError, d_interm_results);
    VALIDATE_AND_RELEASE(cudaError, d_result);
    VALIDATE_CUDA
    (
        cudaError,
        "Failed to execution '_d_minReduce' kernel",
        h_result
    );

    // Starting '_d_minReduce' kernel again with one block to get final results
    _d_minReduce
    <<<
        1, chunk_size, sizeof(result_t) * chunk_size
    >>>(chunk_size, d_interm_results, d_result);
    cudaDeviceSynchronize();
    cudaError = cudaGetLastError();
    WRITE_STATUS(cudaStatus, cudaError);
    VALIDATE_AND_RELEASE(cudaError, d_interm_results);
    VALIDATE_AND_RELEASE(cudaError, d_result);
    VALIDATE_CUDA
    (
        cudaError, 
        "Failed to execution '_d_minReduce' kernel",
        h_result
    );

    // Copy final result from device to host
    cudaError = cudaMemcpy
    (
        &h_result, d_result,
        sizeof(result_t),
        cudaMemcpyDeviceToHost
    );
    WRITE_STATUS(cudaStatus, cudaError);
    VALIDATE_AND_RELEASE(cudaError, d_interm_results);
    VALIDATE_AND_RELEASE(cudaError, d_result);
    VALIDATE_CUDA
    (
        cudaError,
        "Failed to copy memory from device to host.",
        h_result
    );

    // Release all device memory and return the result
    SAFE_RELEASE_CUDA_MEM(d_interm_results);
    SAFE_RELEASE_CUDA_MEM(d_result);
    return h_result;
}

__host__ result_t device_solve_task
(
    const Device& dev,
    num_t n,
    const dim3& globalDim3,
    cudaError_t* const cudaStatus
)
{
    cudaError_t cudaError;
    // Initialize h_result with an invalid value
    result_t h_result = {std::numeric_limits<num_t>::max(), 0, 0};
    result_t* d_results;

    unsigned int max_threads  = dev.getMaxThreadsDim()[0];
    unsigned int warp_size    = dev.getWarpSize();
    unsigned int blocks_count = globalDim3.x;

    cudaError = cudaMalloc(&d_results, sizeof(result_t) * blocks_count);
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
        n += warp_size * blocks_count
    )
    {
        // Starting main kernel. Each CUDA block find the minimum value of "n"
        // in it's numbers group and writes result to global memory
        _d_solve_task
        <<<
            blocks_count,
            {max_threads / warp_size, warp_size},
            sizeof(result_t) * warp_size
        >>>(n, d_results);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        WRITE_STATUS(cudaStatus, cudaError);
        VALIDATE_AND_RELEASE(cudaError, d_results);
        VALIDATE_CUDA
        (
            cudaError, "Failed to run '_d_solve_task' kernel.", h_result
        );

        // Find the minimum value for the num field among the
        // intermediate results
        h_result = _h_minReduce(dev, blocks_count, d_results, &cudaError);
        WRITE_STATUS(cudaStatus, cudaError);
        VALIDATE_AND_RELEASE(cudaError, d_results);
        VALIDATE_CUDA_NO_PRINT(cudaError, h_result);
    }

    // Release all device memory and return the result
    SAFE_RELEASE_CUDA_MEM(d_results);
    return h_result;
}
///////////////////////////////////////////////////////////////////////////////

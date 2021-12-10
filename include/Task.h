#ifndef TASK_H
#define TASK_H


#include <array>

#include "Common.h"
#include "Solver.cuh"
#include "Timer.h"


class Task
{
public:
    using dim3_t = std::array<size_t, 3>;

private:

    // input number
    num_t m_N;

	// results
	result_t m_resultCPU;
	result_t m_resultGPU;

    // Size of dimensions of blocks
    dim3 m_globalDim;

public:
    Task() {}
	~Task() {}

    inline void setData(num_t n) { m_N = n; }
    inline void setGlobalDim(const dim3_t& globalDim)
    {
        m_globalDim.x = globalDim[0];
        m_globalDim.y = globalDim[1];
        m_globalDim.z = globalDim[2];
    }

    inline bool CPU_Solve_task()
    { 
        m_resultCPU = h_solve_task(m_N);
        return true;
    }
	inline bool GPU_Solve_task()
    {
        cudaError_t cudaError;
        
        m_resultGPU = h_start_kernel(m_N, m_globalDim, &cudaError);
        VALIDATE_CUDA_NO_PRINT(cudaError, false);

        return true;
    }

	inline bool ValidateCPU() const
    { 
        return m_resultCPU.base > 1 &&
               h_d_checkPower(m_resultCPU.sum, m_resultCPU.base); 
    }
    bool ValidateGPU() const
    { 
        return m_resultGPU.base > 1 &&
               h_d_checkPower(m_resultGPU.sum, m_resultGPU.base); 
    }
    inline bool ValidateResults() const { return m_resultCPU == m_resultGPU; }

    bool TestCPU();
    bool TestGPU();

    inline result_t getResultCPU() const { return m_resultCPU; }
    inline result_t getResultGPU() const { return m_resultGPU; }
    
};  // Task


#endif // TASK_H

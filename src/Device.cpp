#include "Device.h"

#include <iostream>


Device::Device(cudaError_t* const cudaStatus)
{
    cudaError_t cudaError;
    
    cudaError = cudaGetDevice(&m_handle);
    WRITE_STATUS(cudaStatus, cudaError);
    VALIDATE_CUDA(cudaError, "Failed to get device.",);

    cudaError = cudaGetDeviceProperties(&m_properties, m_handle);
    WRITE_STATUS(cudaStatus, cudaError);
    VALIDATE_CUDA(cudaError, "Failed to get device properties.",);
}

Device::Device(int handle, cudaError_t* const cudaStatus)
{
    cudaError_t cudaError;
    
    m_handle = handle;

    cudaError = cudaGetDeviceProperties(&m_properties, m_handle);
    WRITE_STATUS(cudaStatus, cudaError);
    VALIDATE_CUDA(cudaError, "Failed to get device properties.",);
}


std::ostream& operator<<(std::ostream& stream, const Device& dev)
{    
    stream << "Device information:" << std::endl;
    stream << "Name: " << dev.getName() << "." << std::endl;
    stream << "Global memory size: " << dev.getGlobalMem()
        << " bytes." << std::endl;
    stream << "Shared memory size per block: " <<
        dev.getSharedPerBlockMem() << " bytes." << std::endl;
    stream << "Registers per block: "
        << dev.getRegistersPerBlock() << "." << std::endl;
    stream << "Warp size: "
        << dev.getWarpSize() << "." << std::endl;
    auto t_dims = dev.getMaxThreadsDim();
    stream << "Threads dimensions: " << "<<<" << t_dims[0] << ", "
        << t_dims[1] << ", " << t_dims[2] << ">>>." << std::endl;
    auto g_dims = dev.getMaxGridSize();
    stream << "Grid dimensions: " << "<<<" << g_dims[0] << ", "
        << g_dims[1] << ", " << g_dims[2] << ">>>." << std::endl;
    
    return stream;
}

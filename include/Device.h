#ifndef DEVICE_H
#define DEVICE_H


#include "Common.h"

#include <array>
#include <string>
#include <ostream>


class Device
{
private:
    int m_handle;
    cudaDeviceProp m_properties;

public:
    Device(cudaError_t* const cudaStatus = nullptr);
    Device(int handle, cudaError_t* const cudaStatus = nullptr);
    Device(const Device& other) = default;
    Device(Device&& other) = default;
    ~Device() {}

    Device& operator=(const Device& other) = default;
    Device& operator=(Device&& other) = default;

    inline bool operator==(const Device& other)
    { return m_handle == other.m_handle; }
    inline bool operator!=(const Device& other)
    { return !(*this == other); }


    inline std::string getName() const
    { return m_properties.name; }
    inline size_t getGlobalMem() const
    { return m_properties.totalGlobalMem; }
    inline size_t getSharedPerBlockMem() const
    { return m_properties.sharedMemPerBlock; }
    inline unsigned int getRegistersPerBlock() const
    { return m_properties.regsPerBlock; }
    inline unsigned int getWarpSize() const
    { return m_properties.warpSize; }
    inline std::array<unsigned int, 3> getMaxThreadsDim() const
    {
        auto t_dim = m_properties.maxThreadsDim;
        return
        { 
            static_cast<unsigned int>(t_dim[0]),
            static_cast<unsigned int>(t_dim[1]),
            static_cast<unsigned int>(t_dim[2])
        };
    }
    inline std::array<unsigned int, 3> getMaxGridSize() const
    {
        auto g_dim = m_properties.maxGridSize;
        return
        { 
            static_cast<unsigned int>(g_dim[0]),
            static_cast<unsigned int>(g_dim[1]),
            static_cast<unsigned int>(g_dim[2])
        };
    }

    inline cudaDeviceProp getProp() const
    { return m_properties; }
    inline int getHandle()
    { return m_handle; }

    inline cudaError_t setToExecution() const
    { return cudaSetDevice(m_handle); }

};  // class Device


std::ostream& operator<<(std::ostream& stream, const Device& dev);


#endif // DEVICE_H

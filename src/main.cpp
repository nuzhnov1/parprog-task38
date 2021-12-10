#include "ConsoleApplication.h"
#include "Device.h"


int main(int argc, char* argv[])
{
    cudaError_t cudaError;
    Device dev(&cudaError);

    VALIDATE_CUDA_NO_PRINT(cudaError, -1);

    ConsoleApplication::mainLoop(dev, argc, argv);
}

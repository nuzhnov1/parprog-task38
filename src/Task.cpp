#include "Task.h"

#include <iostream>
#include <iomanip>


bool Task::TestCPU()
{
    Timer timer;

    timer.start();
    if (!CPU_Solve_task()) return false;
    timer.stop();

    std::cout << std::setprecision(3) << "Time of solving task on CPU: " <<
        timer.GetElapsedSeconds() << " seconds." << std::endl; 

    return true;
}

bool Task::TestGPU()
{
	Timer timer;

    timer.start();
    if (!GPU_Solve_task()) return false;
    timer.stop();

    std::cout << std::setprecision(3) << "Time of solving task on GPU: " <<
        timer.GetElapsedSeconds() << " seconds." << std::endl;

    return true; 
}

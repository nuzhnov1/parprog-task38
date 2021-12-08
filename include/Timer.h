#ifndef TIMER_H
#define TIMER_H


#ifdef _WIN32
#include <Windows.h>
#elif defined (__APPLE__) || defined(MACOSX)
#include <sys/time.h>
#else
#include <sys/time.h>
#include <time.h>
#endif


// Timer
class Timer
{
private:
#ifdef WIN32
    LARGE_INTEGER		m_StartTime;
    LARGE_INTEGER		m_EndTime;
#else
    struct timeval		m_start;
    struct timeval		m_stop;
#endif

public:
    // Default constructor
    Timer() {}
    // Copy constructor
    Timer(const Timer& other) = default;
    // Move constructor
    Timer(Timer&& other) = default;
    // Destructor
    ~Timer() {}

    // Copy assigment operator
    Timer& operator=(const Timer& other) = default;
    // Move assigment operator
    Timer& operator=(Timer&& other) = default;

    // Set start timestamp
    inline void start()
    {
#ifdef _WIN32
        QueryPerformanceCounter(&m_StartTime);
#else
        gettimeofday(&m_start, nullptr);
#endif
    }

    // Set stop timestamp
    inline void stop()
    {
#ifdef _WIN32
        QueryPerformanceCounter(&m_EndTime);
#else
        gettimeofday(&m_stop, nullptr);
#endif
    }

    // Get elapsed time between stop and start time moments
    
    inline double GetElapsedMicroseconds() const
    {
        double delta = 
            ((double)(m_stop.tv_sec) * 1.0e+6 + (double)(m_stop.tv_usec)) - 
            ((double)(m_start.tv_sec) * 1.0e+6 + (double)(m_start.tv_usec));
        return delta;
    }

    inline double GetElapsedMilliseconds() const
    { return GetElapsedMicroseconds() / 1.0e+3; }
    inline double GetElapsedSeconds() const
    { return GetElapsedMicroseconds() / 1.0e+6; }

};  // Class Timer


#endif // TIMER_H

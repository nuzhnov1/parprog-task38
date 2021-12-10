#ifndef CONSOLE_APPLICATION_H
#define CONSOLE_APPLICATION_H


#include <string>
#include <cstdint>
#include <limits>

#include <tclap/CmdLine.h>

#include "Task.h"
#include "Device.h"


class ConsoleApplication
{
private:
    struct Argumets
    {
        // Flags bitfield
        bool c: 1; bool g: 1; bool i: 1; bool v: 1;

        unsigned int block_size;
        size_t n_value;
    };

    static Task s_task;
    static Argumets s_args;

public:
    static constexpr num_t MIN_NUM = 0;
    // std::numeric_limits<num_t>::max() - is invalid value
    static constexpr num_t MAX_NUM = std::numeric_limits<num_t>::max() - 1;
    static constexpr unsigned int DEFAULT_BLOCKS = 1024;

    static int mainLoop(const Device& dev, int argc, char* argv[]);

    static bool parseArguments(int argc, char* argv[]);
    static bool solve();
    static void printResult(const result_t& result);

};  // Class ConsoleApplication 


#endif  // CONSOLE_APPLICATION_H

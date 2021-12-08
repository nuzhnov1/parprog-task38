#ifndef CONSOLE_APPLICATION_H
#define CONSOLE_APPLICATION_H


#include <string>
#include <cstdint>
#include <limits>

#include <tclap/CmdLine.h>

#include "Task.h"


class ConsoleApplication
{
private:
    struct Argumets
    {
        // Flags bitfield
        bool c: 1; bool g: 1;

        unsigned int block_size;
        size_t n_value;
    };

    static Task s_task;
    static Argumets s_args;

public:
    static constexpr num_t MIN_NUM = 0;
    static constexpr num_t MAX_NUM = std::numeric_limits<num_t>::max() - 1;
    static constexpr unsigned int MIN_BLOCKS = 1;
    static constexpr unsigned int MAX_BLOCKS = 2048;
    static constexpr unsigned int DEFAULT_BLOCKS = 1024;

    static int mainLoop(int argc, char* argv[]);

    static bool parseArguments(int argc, char* argv[]);
    static bool solve();
    static void printResult(const result_t& result);

};  // Class ConsoleApplication 


#endif  // CONSOLE_APPLICATION_H

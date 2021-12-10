#include "ConsoleApplication.h"

#include <iostream>
#include <cstdlib>
#include <utility>
#include <cmath>


#define CHECK_ERROR(error, ret) if ((error) != cudaSuccess) return (ret)
#define CHECK_CALL(success, ret) if (!(success)) return (ret)


Task ConsoleApplication::s_task{};
ConsoleApplication::Argumets ConsoleApplication::s_args{};


constexpr const char* DESCRIPTION = 
"task38. Finds for a given number such a number greater than the given one "
"that the sum of compact subsequences of digits of this number is a power of "
"another integer, while the power is greater than 1. For example, for the "
"number 1234, compact subsequences will be: 1, 2, 3, 4, 12, 23, 34, 123 and "
"234.";


// Checks the input value of number
class NumberConstraint: public TCLAP::Constraint<num_t>
{
public:
    NumberConstraint(): TCLAP::Constraint<num_t>() {}
    virtual ~NumberConstraint() {}

    virtual bool check(const num_t& value) const override
    {
        return
        (
            value >= ConsoleApplication::MIN_NUM &&
            value <= ConsoleApplication::MAX_NUM
        );
    }

    virtual std::string description() const override
    {
        std::string desc = 
            std::string("input value of number must not be greater then ") +
            std::to_string(ConsoleApplication::MAX_NUM) + " or less then " +
            std::to_string(ConsoleApplication::MIN_NUM);
        
        return desc;
    }

    virtual std::string shortID() const override
    {
        return std::string("input value of number");
    }
};

// Checks the input value of blocks
class BlocksConstraint: public TCLAP::Constraint<unsigned int>
{
private:
    unsigned int m_max_blocks;

public:
    BlocksConstraint(unsigned int max_blocks):
        TCLAP::Constraint<unsigned int>(), m_max_blocks(max_blocks)
    {}
    virtual ~BlocksConstraint() {}

    virtual bool check(const unsigned int& value) const override
    {
        bool is_valid = value >= 1 && value <= m_max_blocks;

        return is_valid;
    }

    virtual std::string description() const override
    {
        std::string desc = 
            std::string("input value of blocks must not be greater then ") +
            std::to_string(m_max_blocks) + " or less then 1";
        
        return desc;
    }

    virtual std::string shortID() const override
    {
        return std::string("input value of blocks");
    }
};


int ConsoleApplication::mainLoop(const Device& dev, int argc, char* argv[])
{
    s_task.setDevice(dev);
    CHECK_CALL(parseArguments(argc, argv), -2);
    
    // If "-i" argument is set - print information about device
    if (s_args.i)
        std::cout << dev << std::endl;
    
    // If value of number is set
    if (s_args.v)
    {
        // And the flags of the using methods are not set
        if (!s_args.c && !s_args.g)
            // By default use "-g" flag and solve task only on GPU
            s_args.g = true;
    }
    // If value of number is not set, but one of the flags "-c" or "-g" is set
    else if (s_args.c || s_args.g)
    {
        std::cerr << "Error: value argument is not specified." << std::endl;
        return -3; 
    }
    
    CHECK_CALL(solve(), -4);

    return 0;
}

bool ConsoleApplication::parseArguments(int argc, char* argv[])
{
    TCLAP::CmdLine cmd(DESCRIPTION, ' ', "1.0");

    try
    {
        TCLAP::SwitchArg c_arg
        (
            "c", "use-cpu", "Solve task on CPU", false
        );
        TCLAP::SwitchArg g_arg
        (
            "g", "use-gpu",
            "Solve task on GPU, using CUDA", false
        );
        TCLAP::SwitchArg i_arg
        (
            "i", "info",
            "Print information about using devices", false
        );
        
        BlocksConstraint blocks_constraint
        (
            s_task.getDevice().getMaxGridSize()[0]
        );
        TCLAP::ValueArg<unsigned int> b_arg
        (
            "b",  "blocks",
            "Number of CUDA blocks using to parallel", false, DEFAULT_BLOCKS, 
            &blocks_constraint,
            nullptr
        );

        NumberConstraint num_constraint;
        TCLAP::UnlabeledValueArg<num_t> val_arg
        (
            "v", "input value",
            false, 0,
            &num_constraint, false,
            nullptr 
        );
        
        cmd.add(val_arg);
        cmd.add(b_arg);
        cmd.add(i_arg); cmd.add(g_arg); cmd.add(c_arg);

        cmd.parse(argc, argv);

        s_args.v = val_arg.isSet();
        s_args.c = c_arg.isSet();
        s_args.g = g_arg.isSet();
        s_args.i = i_arg.isSet();
        s_args.block_size = b_arg.getValue();
        s_args.n_value = val_arg.getValue();
    }
    catch (TCLAP::ArgException& exception)
    {
        std::cerr << "Failed parse of argument: " << exception.what() <<
            "." << std::endl;
        return false;
    }
    
    return true;
}

bool ConsoleApplication::solve()
{
    s_task.setData(s_args.n_value);

    // if "g" flag is set - initialize value of dimensions
    if (s_args.g)
        s_task.setGlobalDim({s_args.block_size, 1, 1});

    if (s_args.c)
    {
        std::cout << "Solving task on CPU..." << std::endl;
        CHECK_CALL(s_task.TestCPU(), false);

        if(!s_task.ValidateCPU())
        {
            std::cerr << "Failed to solving task on CPU: "
                "invalid sum or power values!" << std::endl;
            
            return false;
        }

        std::cout << "Result of solving task on CPU:" << std::endl;
        printResult(s_task.getResultCPU());
    }

    if (s_args.g)
    {
        std::cout << "Solving task on GPU..." << std::endl;
        CHECK_CALL(s_task.TestGPU(), false);

        if(!s_task.ValidateGPU())
        {
            std::cerr << "Failed to solving task on GPU: "
                "invalid sum or power values!" << std::endl;
            
            return false;
        }

        std::cout << "Result of solving task on GPU:" << std::endl;
        printResult(s_task.getResultGPU());
    }

    if (s_args.c && s_args.g)
    {
        if (!s_task.ValidateResults())
        {
            std::cerr << "The results of solving the task on the "
                "CPU and GPU are different!" << std::endl;
            
            return false;
        }
        else
        {
            std::cerr << "The results of solving the task on the "
                "CPU and GPU are the same." << std::endl;
        }
    }

    return true;
}

void ConsoleApplication::printResult(const result_t& result)
{
    std::cout << "Number value: " << result.num << "." << std::endl;
    std::cout << "Sum of subsequences: " << result.sum << "." << std::endl;
    std::cout << "Base: " << result.base << "." << std::endl;
    std::cout << result.base << " ^ " <<
        (num_t)round(log(result.sum) / log(result.base)) << " = " <<
        result.sum << "." << std::endl;
}

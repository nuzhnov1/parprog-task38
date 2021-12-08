# Variable definitions
MKDIR   := mkdir -p
RMDIR	:= rm -rf
COPY	:= cp
TAR		:= tar -cf

# Using compilers
CC	:= g++
NV	:= nvcc

# Compiler flags
CFLAGS  := -O3 -std=c++11 -Wall -Wpedantic
NVFLAGS := -O3 -std c++11 -m 64 --expt-relaxed-constexpr --use_fast_math

# Common directories
BIN             := ./bin
LIB             := ./lib
OBJ             := ./obj
INCLUDE         := ./include
SRC             := ./src

# Varriables for program
PROG_NAME   := task38

PROG_SRCS   := $(wildcard $(SRC)/*.cpp)
CUDA_SRCS   := $(wildcard $(SRC)/*.cu)
PROG_OBJS   := $(patsubst $(SRC)/%.cpp,$(OBJ)/%.o,$(PROG_SRCS))
CUDA_OBJS	:= $(patsubst $(SRC)/%.cu,$(OBJ)/%.o,$(CUDA_SRCS))
PROG_BIN    := $(BIN)/$(PROG_NAME)


# Phony targets
.PHONY: program debug clean tar


# Default target
all: program


# Build program target
program: $(PROG_BIN)
	$(info Building a program is complete. Executable file is located \
	in "$(BIN)" directory.)

# Debug target
debug: CFLAGS	:= -g -std=c++11 -Wall -Wpedantic
debug: NVFLAGS	:= -g -G -std c++11 -m 64 --expt-relaxed-constexpr
debug: program

# Clean target
clean:
	$(info Removing a directories "$(OBJ)" and "$(BIN)"...)
	$(RMDIR) $(OBJ) $(BIN)

# Create "tar" target
tar:
	$(info Archiving the project...)
	$(TAR) $(PROG_NAME).tar $(INCLUDE) $(SRC) Makefile
	$(info Project has archived. Archive file: $(PROG_NAME).tar)


# Creating directories target
$(BIN) $(OBJ):
	$(info Creating a directory "$@"...)
	$(MKDIR) $@


# Compilation cpp files target
$(OBJ)/%.o: $(SRC)/%.cpp | $(OBJ)
	$(info Compiling a "$<" file...)
	$(CC) $(CFLAGS) -I$(INCLUDE) -c $< -o $@

# Compilation cuda files target
$(OBJ)/%.o: $(SRC)/%.cu | $(OBJ)
	$(info Compiling a "$<" file...)
	$(NV) $(NVFLAGS) -I$(INCLUDE) -c $< -o $@

# Linkage program target
$(PROG_BIN): $(PROG_OBJS) $(CUDA_OBJS) | $(BIN)
	for item in $^ ; do \
		echo "Linking a $$item file..." ; \
	done
	$(CC) $^ -o $@ -lcuda -lcudart

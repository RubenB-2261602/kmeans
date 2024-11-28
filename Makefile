# -std=c++14: we're limiting ourselves to c++14, since that's what the 
#             GCC compiler on the VSC supports.
# -DNDEBUG: turns off e.g. assertion checks
# -O3: enables optimizations in the compiler

# Settings for optimized build
FLAGS=-O3 -DNDEBUG -std=c++14

# Settings for a debug build
#FLAGS=-g -std=c++14

ifeq ($(OS), Windows_NT)
    RM = del
    EXT = .exe
else
    RM = rm
    EXECUTABLE = 
endif

all: kmeans

clean:
	$(RM) kmeans$(EXT)

kmeans: main_startcode.cpp rng.cpp kmeans.cu
	nvcc $(FLAGS) -o kmeans$(EXT) main_startcode.cpp rng.cpp kmeans.cu

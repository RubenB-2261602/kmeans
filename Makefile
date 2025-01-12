CXX=mpic++
CXXFLAGS=-O3 -Wall -std=c++17

all: kmeans_mpi

kmeans_mpi: main_startcode.cpp CSVReader.hpp CSVWriter.hpp rng.h timer.h
	$(CXX) $(CXXFLAGS) -o kmeans_mpi main_startcode.cpp

clean:
	rm -f kmeans_mpi

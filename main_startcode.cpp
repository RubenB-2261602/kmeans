#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <mpi.h>
#include "CSVReader.hpp"
#include "CSVWriter.hpp"
#include "rng.h"
#include "timer.h"

void usage()
{
	std::cerr << R"XYZ(
Usage:

  kmeans --input inputfile.csv --output outputfile.csv --k numclusters --repetitions numrepetitions --seed seed [--blocks numblocks] [--threads numthreads] [--trace clusteridxdebug.csv] [--centroidtrace centroiddebug.csv]

Arguments:

 --input:
 
   Specifies input CSV file, number of rows represents number of points, the
   number of columns is the dimension of each point.

 --output:

   Output CSV file, just a single row, with as many entries as the number of
   points in the input file. Each entry is the index of the cluster to which
   the point belongs. The script 'visualize_clusters.py' can show this final
   clustering.

 --k:

   The number of clusters that should be identified.

 --repetitions:

   The number of times the k-means algorithm is repeated; the best clustering
   is kept.

 --blocks:

   Only relevant in CUDA version, specifies the number of blocks that can be
   used.

 --threads:

   Not relevant for the serial version. For the OpenMP version, this number 
   of threads should be used. For the CUDA version, this is the number of 
   threads per block. For the MPI executable, this should be ignored, but
   the wrapper script 'mpiwrapper.sh' can inspect this to run 'mpirun' with
   the correct number of processes.

 --seed:

   Specifies a seed for the random number generator, to be able to get 
   reproducible results.

 --trace:

   Debug option - do NOT use this when timing your program!

   For each repetition, the k-means algorithm goes through a sequence of 
   increasingly better cluster assignments. If this option is specified, this
   sequence of cluster assignments should be written to a CSV file, similar
   to the '--output' option. Instead of only having one line, there will be
   as many lines as steps in this sequence. If multiple repetitions are
   specified, only the results of the first repetition should be logged
   for clarity. The 'visualize_clusters.py' program can help to visualize
   the data logged in this file.

 --centroidtrace:

   Debug option - do NOT use this when timing your program!

   Should also only log data during the first repetition. The resulting CSV 
   file first logs the randomly chosen centroids from the input data, and for
   each step in the sequence, the updated centroids are logged. The program 
   'visualize_centroids.py' can be used to visualize how the centroids change.
   
)XYZ";
	exit(-1);
}

// Helper function to read input file into allData, setting number of detected
// rows and columns. Feel free to use, adapt or ignore
void readData(std::ifstream& input, std::vector<double>& allData, size_t& numRows, size_t& numCols)
{
	if (!input.is_open())
		throw std::runtime_error("Input file is not open");

	allData.resize(0);
	numRows = 0;
	numCols = -1;

	CSVReader inReader(input);
	int numColsExpected = -1;
	int line = 1;
	std::vector<double> row;

	while (inReader.read(row))
	{
		if (numColsExpected == -1)
		{
			numColsExpected = row.size();
			if (numColsExpected <= 0)
				throw std::runtime_error("Unexpected error: 0 columns");
		}
		else if (numColsExpected != (int)row.size())
			throw std::runtime_error("Incompatible number of colums read in line " + std::to_string(line) + ": expecting " + std::to_string(numColsExpected) + " but got " + std::to_string(row.size()));

		for (auto x : row)
			allData.push_back(x);

		line++;
	}

	numRows = (size_t)allData.size() / numColsExpected;
	numCols = (size_t)numColsExpected;
}

FileCSVWriter openDebugFile(const std::string& n)
{
	FileCSVWriter f;

	if (n.length() != 0)
	{
		f.open(n);
		if (!f.is_open())
			std::cerr << "WARNING: Unable to open debug file " << n << std::endl;
	}
	return f;
}

std::vector<std::vector<double>> makeCentroids(const std::vector<double>& allData, const std::vector<size_t>& indices, size_t numCols)
{
	std::vector<std::vector<double>> centroids(indices.size());
	for (size_t i = 0; i < indices.size(); i++)
	{
		centroids[i].resize(numCols);
		for (size_t j = 0; j < numCols; j++)
			centroids[i][j] = allData[indices[i] * numCols + j];
	}
	return centroids;

}

std::pair<double, int> find_closest_centroid_index_and_distance(const std::vector<double>& point, const std::vector<std::vector<double>>& centroids)
{
	double minDistance = std::numeric_limits<double>::max();
	int centroidIndex = 0;
	for (int i = 0; i < centroids.size(); i++)
	{
		double distance = 0.0;
		for (int j = 0; j < point.size(); j++)
		{
			distance += std::pow(point[j] - centroids[i][j], 2);
		}
		if (distance < minDistance)
		{
			centroidIndex = i;
			minDistance = distance;
		}
	}
	return { minDistance, centroidIndex };
}

std::vector<double> average_of_points_with_cluster(int clusterIndex, const std::vector<int>& clusters, const std::vector<double>& allData, size_t numCols)
{
	double average = 0;
	int count = 0;
	std::vector<double> averageVector(numCols);

	for (size_t i = 0; i < clusters.size(); i++)
	{
		if (clusters[i] == clusterIndex)
		{
			for (int j = 0; j < numCols; j++)
			{
				int index = i * numCols + j;
				averageVector[j] += allData[index];
			}
			++count;
		}
	}

	// divide by count
	for (int i = 0; i < numCols; i++)
	{
		averageVector[i] /= count;
	}

	return averageVector;
}

int kmeans(Rng& rng, const std::string& inputFile, const std::string& outputFileName,
	int numClusters, int repetitions, int numBlocks, int numThreads,
	const std::string& centroidDebugFileName, const std::string& clusterDebugFileName)
{
	FileCSVWriter centroidDebugFile = openDebugFile(centroidDebugFileName);
	FileCSVWriter clustersDebugFile = openDebugFile(clusterDebugFileName);
	FileCSVWriter csvOutputFile(outputFileName);

	int rank, size, len;
	char name[MPI_MAX_PROCESSOR_NAME + 1];
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name, &len);

	std::cout << "Ready: " << name << " " << rank << " of " << size << std::endl;

	std::vector<double> allData;
	size_t numRows = 0, numCols = 0;

	// Only rank 0 reads the data
	if (rank == 0)
	{
		if (!csvOutputFile.is_open())
		{
			std::cerr << "Unable to open output file " << outputFileName << std::endl;
			MPI_Abort(MPI_COMM_WORLD, -1);
			return -1;
		}

		std::ifstream input(inputFile);
		if (!input.is_open())
		{
			std::cerr << "Unable to open input file " << inputFile << std::endl;
			MPI_Abort(MPI_COMM_WORLD, -1);
			return -1;
		}
		readData(input, allData, numRows, numCols);
	}

	// Broadcast dimensions to all processes
	MPI_Bcast(&numRows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
	MPI_Bcast(&numCols, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

	// Calculate points per process and remainder
	int pointsPerProcess = numRows / size;
	int remainder = numRows % size;
	int localPoints = pointsPerProcess + (rank < remainder ? 1 : 0);
	int offset = rank * pointsPerProcess + std::min(rank, remainder);

	// Allocate local data arrays
	std::vector<double> localData(localPoints * numCols);
	std::vector<int> localClusters(localPoints, -1);
	std::vector<int> clusters(numRows, -1);

	// Distribute data using MPI_Scatterv
	std::vector<int> sendcounts(size);
	std::vector<int> displs(size);
	if (rank == 0) {
		for (int i = 0; i < size; i++) {
			sendcounts[i] = (pointsPerProcess + (i < remainder ? 1 : 0)) * numCols;
			displs[i] = (i * pointsPerProcess + std::min(i, remainder)) * numCols;
		}
	}

	MPI_Scatterv(allData.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
				localData.data(), localPoints * numCols, MPI_DOUBLE,
				0, MPI_COMM_WORLD);

	Timer timer;
	std::vector<int> bestClusters;
	double bestDistSquaredSum = std::numeric_limits<double>::max();
	std::vector<size_t> stepsPerRepetition(repetitions);

	for (int r = 0; r < repetitions; r++)
	{
		size_t numSteps = 0;
		std::vector<std::vector<double>> centroids(numClusters, std::vector<double>(numCols));
		std::vector<size_t> centroidIndices(numClusters);

		if (rank == 0) {
			rng.pickRandomIndices(numRows, centroidIndices);
			for (int i = 0; i < numClusters; i++) {
				for (size_t j = 0; j < numCols; j++) {
					centroids[i][j] = allData[centroidIndices[i] * numCols + j];
				}
			}
		}

		// Broadcast initial centroids
		for (int i = 0; i < numClusters; i++) {
			MPI_Bcast(centroids[i].data(), numCols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}

		bool globalChanged = true;
		while (globalChanged)
		{
			globalChanged = false;
			double localDistSquaredSum = 0.0;

			// Find closest centroids for local points
			for (int p = 0; p < localPoints; p++)
			{
				std::vector<double> point(numCols);
				for (size_t j = 0; j < numCols; j++) {
					point[j] = localData[p * numCols + j];
				}

				auto [dist, newCluster] = find_closest_centroid_index_and_distance(point, centroids);
				localDistSquaredSum += dist;

				if (newCluster != localClusters[p])
				{
					localClusters[p] = newCluster;
					globalChanged = true;
				}
			}

			// Combine results
			bool anyChanged;
			MPI_Allreduce(&globalChanged, &anyChanged, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
			globalChanged = anyChanged;

			double globalDistSquaredSum;
			MPI_Reduce(&localDistSquaredSum, &globalDistSquaredSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

			// Gather all clusters
			std::vector<int> recvcounts(size);
			std::vector<int> displs(size);
			if (rank == 0) {
				for (int i = 0; i < size; i++) {
					recvcounts[i] = pointsPerProcess + (i < remainder ? 1 : 0);
					displs[i] = i * pointsPerProcess + std::min(i, remainder);
				}
			}

			MPI_Gatherv(localClusters.data(), localPoints, MPI_INT,
						clusters.data(), recvcounts.data(), displs.data(), MPI_INT,
						0, MPI_COMM_WORLD);

			if (globalChanged)
			{
				// Calculate new centroids in parallel
				std::vector<std::vector<double>> localSums(numClusters, std::vector<double>(numCols, 0.0));
				std::vector<int> localCounts(numClusters, 0);

				for (int p = 0; p < localPoints; p++)
				{
					int cluster = localClusters[p];
					localCounts[cluster]++;
					for (size_t j = 0; j < numCols; j++) {
						localSums[cluster][j] += localData[p * numCols + j];
					}
				}

				// Reduce sums and counts
				std::vector<std::vector<double>> globalSums(numClusters, std::vector<double>(numCols, 0.0));
				std::vector<int> globalCounts(numClusters, 0);

				for (int i = 0; i < numClusters; i++) {
					MPI_Allreduce(localSums[i].data(), globalSums[i].data(), numCols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
					MPI_Allreduce(&localCounts[i], &globalCounts[i], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
				}

				// Update centroids
				for (int i = 0; i < numClusters; i++) {
					if (globalCounts[i] > 0) {
						for (size_t j = 0; j < numCols; j++) {
							centroids[i][j] = globalSums[i][j] / globalCounts[i];
						}
					}
				}
			}

			if (rank == 0)
			{
				if (globalDistSquaredSum < bestDistSquaredSum)
				{
					bestDistSquaredSum = globalDistSquaredSum;
					bestClusters = clusters;
				}

				if (r == 0 && numSteps == 0)
				{
					clustersDebugFile.write(clusters, "# Clusters:\n");
				}

				centroidDebugFile.write(centroids, "# Centroids:\n");
			}

			numSteps++;
		}

		if (rank == 0)
		{
			stepsPerRepetition[r] = numSteps;
		}
	}

	timer.stop();

	if (rank == 0)
	{
		std::cerr << "# Type,blocks,threads,file,seed,clusters,repetitions,bestdistsquared,timeinseconds" << std::endl;
		std::cout << "mpi," << numBlocks << "," << numThreads << "," << inputFile << ","
			<< rng.getUsedSeed() << "," << numClusters << ","
			<< repetitions << "," << bestDistSquaredSum << "," << timer.durationNanoSeconds() / 1e9
			<< std::endl;

		csvOutputFile.write(stepsPerRepetition, "# Steps: ");
		csvOutputFile.write(bestClusters);
	}

	MPI_Finalize();
	return 0;
}

int mainCxx(const std::vector<std::string>& args)
{
	if (args.size() % 2 != 0)
		usage();

	std::string inputFileName, outputFileName, centroidTraceFileName, clusterTraceFileName;
	unsigned long seed = 0;

	int numClusters = -1, repetitions = -1;
	int numBlocks = 1, numThreads = 1;
	for (int i = 0; i < args.size(); i += 2)
	{
		if (args[i] == "--input")
			inputFileName = args[i + 1];
		else if (args[i] == "--output")
			outputFileName = args[i + 1];
		else if (args[i] == "--centroidtrace")
			centroidTraceFileName = args[i + 1];
		else if (args[i] == "--trace")
			clusterTraceFileName = args[i + 1];
		else if (args[i] == "--k")
			numClusters = stoi(args[i + 1]);
		else if (args[i] == "--repetitions")
			repetitions = stoi(args[i + 1]);
		else if (args[i] == "--seed")
			seed = stoul(args[i + 1]);
		else if (args[i] == "--blocks")
			numBlocks = stoi(args[i + 1]);
		else if (args[i] == "--threads")
			numThreads = stoi(args[i + 1]);
		else
		{
			std::cerr << "Unknown argument '" << args[i] << "'" << std::endl;
			return -1;
		}
	}

	if (inputFileName.length() == 0 || outputFileName.length() == 0 || numClusters < 1 || repetitions < 1 || seed == 0)
		usage();

	Rng rng(seed);

	return kmeans(rng, inputFileName, outputFileName, numClusters, repetitions,
		numBlocks, numThreads, centroidTraceFileName, clusterTraceFileName);
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	std::vector<std::string> args;
	for (int i = 1; i < argc; i++)
		args.push_back(argv[i]);

	return mainCxx(args);
}
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cuda_runtime.h>

__global__ void assignPointsToCentroids(
    const double *points,
    const double *centroids,
    double *distances,
    int *centroidIndices,
    int numPoints,
    int numCentroids,
    int numCols)
{
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Handle the grid-stride loop to ensure threads handle multiple points
    for (int i = pointIdx; i < numPoints; i += gridDim.x * blockDim.x)
    {
        double minDistance = 1.0e30; // Use a large constant value
        int closestCentroidIdx = 0;

        for (int j = 0; j < numCentroids; j++)
        {
            double distance = 0.0;
            for (int k = 0; k < numCols; k++)
            {
                double diff = points[i * numCols + k] - centroids[j * numCols + k];
                distance += diff * diff;
            }

            distance = sqrt(distance); // Euclidean distance calculation

            if (distance < minDistance)
            {
                minDistance = distance;
                closestCentroidIdx = j;
            }
        }

        // Store the results in the arrays
        distances[i] = minDistance;
        centroidIndices[i] = closestCentroidIdx;
    }
}

void calculateDistance(const std::vector<double> &points,
                       const std::vector<std::vector<double>> &centroids,
                       std::vector<std::pair<double, int>> &distancesAndIndex,
                       int numBlocks, int numThreads)
{
    int numPoints = points.size() / centroids[0].size();
    int numCols = centroids[0].size();
    int numCentroids = centroids.size();

    // Flatten de 2D vectoren naar 1D arrays voor CUDA
    std::vector<double> flatCentroids;
    for (const auto &centroid : centroids)
    {
        flatCentroids.insert(flatCentroids.end(), centroid.begin(), centroid.end());
    }

    double *d_points, *d_centroids, *d_distances;
    int *d_centroidIndices;

    // Alloceren van device geheugen
    cudaMalloc(&d_points, points.size() * sizeof(double));
    cudaMalloc(&d_centroids, flatCentroids.size() * sizeof(double));
    cudaMalloc(&d_distances, numPoints * sizeof(double));
    cudaMalloc(&d_centroidIndices, numPoints * sizeof(int));

    // Gegevens kopiëren naar device
    cudaMemcpy(d_points, points.data(), points.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, flatCentroids.data(), flatCentroids.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Launch de kernel
    int totalThreads = numBlocks * numThreads;
    assignPointsToCentroids<<<numBlocks, numThreads>>>(
        d_points, d_centroids, d_distances, d_centroidIndices,
        min(totalThreads, numPoints), numCentroids, numCols);

    // Synchroniseren van de kernel
    cudaDeviceSynchronize();

    // Controleer op fouten bij de kernel lancering
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error (kernel launch): " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Resultaten terug kopiëren naar de host
    std::vector<double> hostDistances(numPoints);
    std::vector<int> hostCentroidIndices(numPoints);
    cudaMemcpy(hostDistances.data(), d_distances, numPoints * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostCentroidIndices.data(), d_centroidIndices, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

    // Converteer de resultaten terug naar het originele formaat (std::pair)
    distancesAndIndex.resize(numPoints);
    for (int i = 0; i < numPoints; ++i)
    {
        distancesAndIndex[i] = {hostDistances[i], hostCentroidIndices[i]};
    }

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_distances);
    cudaFree(d_centroidIndices);
}

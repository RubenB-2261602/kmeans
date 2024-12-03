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

    // Zorg ervoor dat de thread binnen de geldige range zit
    if (pointIdx < numPoints) {
        double minDistance = 1.0e30; // Use a large constant value
        int closestCentroidIdx = 0;

        for (int i = 0; i < numCentroids; i++) {
            double distance = 0.0;
            for (int j = 0; j < numCols; j++) {
                double diff = points[pointIdx * numCols + j] - centroids[i * numCols + j];
                distance += diff * diff;
            }

            distance = sqrt(distance); // Euclidische afstand berekenen

            if (distance < minDistance) {
                minDistance = distance;
                closestCentroidIdx = i;
            }
        }

        // Resultaten opslaan in de array
        distances[pointIdx] = minDistance;
        centroidIndices[pointIdx] = closestCentroidIdx;
    }
}

void calculateDistance(const std::vector<double>& points, 
    const std::vector<std::vector<double>>& centroids, 
    std::vector<std::pair<double, int>>& distancesAndIndex,
    int numBlocks, int numThreads)
{
    int numPoints = points.size() / centroids[0].size();
    int numCols = centroids[0].size();
    int numCentroids = centroids.size();

    // Flatten de 2D vectoren naar 1D arrays voor CUDA
    std::vector<double> flatCentroids;
    for (const auto& centroid : centroids) {
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
    int numIterations = (numPoints + totalThreads - 1) / totalThreads;

    for (int i = 0; i < numIterations; ++i) {
    int offset = i * totalThreads;
    assignPointsToCentroids<<<numBlocks, numThreads>>>(
    d_points + offset * numCols, d_centroids, d_distances + offset, d_centroidIndices + offset, 
    min(totalThreads, numPoints - offset), numCentroids, numCols
    );
    }

    // Synchroniseren van de kernel
    cudaDeviceSynchronize();

    // Controleer op fouten bij de kernel lancering
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
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
    for (int i = 0; i < numPoints; ++i) {
    distancesAndIndex[i] = {hostDistances[i], hostCentroidIndices[i]};
    }

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_distances);
    cudaFree(d_centroidIndices);
}

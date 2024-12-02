#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cuda_runtime.h>

// Kernel functie om de dichtstbijzijnde centroid te vinden voor elk punt
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
                //printf("%f - %f\n", points[pointIdx * numCols + j], centroids[i * numCols + j]);
                distance += diff * diff;
            }

            // Controleer of de berekende afstand geldig is
            if (distance < 0) {
                printf("Warning: Negative distance computed! Point: %d, Centroid: %d\n", pointIdx, i);
                distance = 0; // Zorg ervoor dat de afstand niet negatief is
            }

            distance = sqrt(distance); // Euclidische afstand berekenen

            // Controleer of sqrt een geldige waarde oplevert
            if (isnan(distance)) {
                printf("NaN computed for Point %d, Centroid %d!\n", pointIdx, i);
                distance = 0; // Voorkom NaN
            }

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


// Functie om de afstand en de bijbehorende centroid van de punten te berekenen
void calculateDistance(std::vector<std::vector<double>>& points, 
                       std::vector<std::vector<double>>& centroids, 
                       std::vector<std::pair<double, int>>& distancesAndIndex,
                       int numBlocks, int numThreads)
{
    int numPoints = points.size();
    int numCols = points[0].size();
    int numCentroids = centroids.size();

    // Flatten de 2D vectoren naar 1D arrays voor CUDA
    std::vector<double> flatPoints;
    for (const auto& point : points) {
        flatPoints.insert(flatPoints.end(), point.begin(), point.end());
    }

    std::vector<double> flatCentroids;
    for (const auto& centroid : centroids) {
        flatCentroids.insert(flatCentroids.end(), centroid.begin(), centroid.end());
    }

    double *d_points, *d_centroids, *d_distances;
    int *d_centroidIndices;

    // Alloceren van device geheugen
    cudaMalloc(&d_points, numPoints * numCols * sizeof(double));
    cudaMalloc(&d_centroids, numCentroids * numCols * sizeof(double));
    cudaMalloc(&d_distances, numPoints * sizeof(double));
    cudaMalloc(&d_centroidIndices, numPoints * sizeof(int));

    // Gegevens kopiëren naar device
    cudaMemcpy(d_points, flatPoints.data(), numPoints * numCols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, flatCentroids.data(), numCentroids * numCols * sizeof(double), cudaMemcpyHostToDevice);

    // Launch de kernel
    int totalThreads = numBlocks * numThreads;
    int numIterations = (numPoints + totalThreads - 1) / totalThreads;

    for (int i = 0; i < numIterations; ++i) {
        int offset = i * totalThreads;
        assignPointsToCentroids<<<numBlocks, numThreads>>>(d_points + offset * numCols, d_centroids, d_distances + offset, d_centroidIndices + offset, min(totalThreads, numPoints - offset), numCentroids, numCols);
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

    // Print de resultaten
    // for (int i = 0; i < numPoints; i++) {
    //     std::cout << "Host output - Point " << i << ": Distance: " << distancesAndIndex[i].first << ", Centroid: " << distancesAndIndex[i].second << std::endl;
    // }

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_distances);
    cudaFree(d_centroidIndices);
}
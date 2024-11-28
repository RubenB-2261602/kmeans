#include "kmeans.cuh"

#include <vector>

__global__ void assignPointstoCentroid(
    const double *points,
    const double *centroids,
    int *distancesAndIndex)
{
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int numPoints = gridDim.x * blockDim.x;
    int numCentroids = gridDim.y * blockDim.y;

    if (pointIdx < numPoints) {
        double minDistance = DBL_MAX;
        int centroidIndex = 0;

        for (int i = 0; i < numCentroids; i++) {
            double distance = 0.0;
            for (int j = 0; j < numPoints; j++) {
                double diff = points[pointIdx * numPoints + j] - centroids[i * numPoints + j];
                distance += diff * diff;
            }
            if (distance < minDistance) {
                minDistance = distance;
                centroidIndex = i;
            }
        }

        distancesAndIndex[pointIdx * 2] = minDistance;
        distancesAndIndex[pointIdx * 2 + 1] = centroidIndex;
    }
}

void calculateDistance(std::vector<std::vector<double>> &points,
                       std::vector<std::vector<double>> &centroids,
                       std::vector<std::pair<double, int>> &distancesAndIndex)
{
    // TODO: Implement this function for cuda

    double *d_points;
    double *d_centroids;
    int *d_distancesAndIndex;

    // Allocate device memory and copy data from host to device
    cudaMalloc(&d_points, points.size() * points[0].size() * sizeof(double));
    cudaMalloc(&d_centroids, centroids.size() * centroids[0].size() * sizeof(double));
    cudaMalloc(&d_distancesAndIndex, distancesAndIndex.size() * sizeof(int));

    cudaMemcpy(d_points, points.data(), points.size() * points[0].size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids.data(), centroids.size() * centroids[0].size() * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    assignPointstoCentroid<<<1, 1>>>(d_points, d_centroids, d_distancesAndIndex);

    // Copy results back to host
    cudaMemcpy(distancesAndIndex.data(), d_distancesAndIndex, distancesAndIndex.size() * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_distancesAndIndex);
}
#include "kmeans.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <limits>
#include <cfloat>
#include <stdexcept>

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

    for (int i = pointIdx; i < numPoints; i += gridDim.x * blockDim.x)
    {
        double minDistance = DBL_MAX;
        int closestCentroidIdx = 0;

        for (int j = 0; j < numCentroids; j++)
        {
            double distance = 0.0;
            for (int k = 0; k < numCols; k++)
            {
                double diff = points[i * numCols + k] - centroids[j * numCols + k];
                distance += diff * diff;
            }
            if (distance < minDistance)
            {
                minDistance = distance;
                closestCentroidIdx = j;
            }
        }

        distances[i] = minDistance;
        centroidIndices[i] = closestCentroidIdx;
    }
}

CudaKMeans::CudaKMeans(size_t numPoints, size_t numCentroids, size_t numCols)
    : numPoints(numPoints), numCentroids(numCentroids), numCols(numCols)
{
    cudaMalloc(&d_points, numPoints * numCols * sizeof(double));
    cudaMalloc(&d_centroids, numCentroids * numCols * sizeof(double));
    cudaMalloc(&d_distances, numPoints * sizeof(double));
    cudaMalloc(&d_centroidIndices, numPoints * sizeof(int));
}

CudaKMeans::~CudaKMeans()
{
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_distances);
    cudaFree(d_centroidIndices);
}

void CudaKMeans::copyPointsToDevice(const std::vector<double> &points)
{
    cudaMemcpy(d_points, points.data(), numPoints * numCols * sizeof(double), cudaMemcpyHostToDevice);
}

void CudaKMeans::copyCentroidsToDevice(const std::vector<std::vector<double>> &centroids)
{
    std::vector<double> flatCentroids;
    for (const auto &centroid : centroids)
        flatCentroids.insert(flatCentroids.end(), centroid.begin(), centroid.end());

    cudaMemcpy(d_centroids, flatCentroids.data(), numCentroids * numCols * sizeof(double), cudaMemcpyHostToDevice);
}

void CudaKMeans::assignCentroids(int numBlocks, int numThreads, std::vector<std::pair<double, int>> &distancesAndIndices)
{
    assignPointsToCentroids<<<numBlocks, numThreads>>>(
        d_points, d_centroids, d_distances, d_centroidIndices, numPoints, numCentroids, numCols);

    std::vector<double> hostDistances(numPoints);
    std::vector<int> hostCentroidIndices(numPoints);

    cudaMemcpy(hostDistances.data(), d_distances, numPoints * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostCentroidIndices.data(), d_centroidIndices, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

    distancesAndIndices.resize(numPoints);
    for (size_t i = 0; i < numPoints; ++i)
    {
        distancesAndIndices[i] = {hostDistances[i], hostCentroidIndices[i]};
    }
}

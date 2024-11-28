#include "kmeans.cuh"

#include <vector>

__global__ void assignPointstoCentroid(
    const double *points,
    const double *centroids,
    const int numPoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPoints)
    {
    }
}

void calculateDistance(std::vector<std::vector<double>> &points,
                       std::vector<std::vector<double>> &centroids,
                       std::vector<std::pair<double, int>> &distancesAndIndex)
{
    // TODO: Implement this function for cude
}
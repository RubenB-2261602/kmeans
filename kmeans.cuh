#ifndef KMEANS_CUH
#define KMEANS_CUH

#include <vector>
#include <cuda_runtime.h>

__global__ void assignPointsToCentroids(
    const double *points,
    const double *centroids,
    double *distances,
    int *centroidIndices,
    int numPoints,
    int numCentroids,
    int numCols);

class CudaKMeans {
public:
    CudaKMeans(size_t numPoints, size_t numCentroids, size_t numCols);
    ~CudaKMeans();

    void copyPointsToDevice(const std::vector<double> &points);
    void copyCentroidsToDevice(const std::vector<std::vector<double>> &centroids);
    void assignCentroids(int numBlocks, int numThreads, std::vector<std::pair<double, int>> &distancesAndIndices);

private:
    size_t numPoints;
    size_t numCentroids;
    size_t numCols;

    double *d_points;
    double *d_centroids;
    double *d_distances;
    int *d_centroidIndices;
};

#endif

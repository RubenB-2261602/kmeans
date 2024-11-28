#ifndef MY_CUDA_FILE_H
#define MY_CUDA_FILE_H

void assignPointstoCentroid(
    const double *points,
    const double *centroids,
    const int numPoints);

void calculateDistance(std::vector<std::vector<double>> &points,
                       std::vector<std::vector<double>> &centroids,
                       std::vector<std::pair<double, int>> &distancesAndIndex)

#endif
#ifndef MY_CUDA_FILE_H
#define MY_CUDA_FILE_H

#include <vector>

void calculateDistance(std::vector<std::vector<double>> &points,
                       std::vector<std::vector<double>> &centroids,
                       std::vector<std::pair<double, int>> &distancesAndIndex,
                       int numBlocks,
                       int numThreads);

#endif
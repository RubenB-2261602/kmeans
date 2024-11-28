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
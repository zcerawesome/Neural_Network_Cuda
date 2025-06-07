#include "matrice_helper.cuh"

void set_data(void* a, void* b, void* c, int n)
{
    cudaMalloc((void**)a, n);
    cudaMalloc((void**)b, n);
    cudaMalloc((void**)c, n);
}

void free_data(void* a, void* b, void* c)
{
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}


template <typename T>
__global__ void General_operation(const T* a, const T* b, T* dest, int n, Operations op)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= n)
        return;
    switch(op)
    {
        case Add:
            dest[idx] = a[idx] + b[idx];
            break;
        case Subtract:
            dest[idx] = a[idx] - b[idx];
            break;
        case Multiply:
            dest[idx] = a[idx] * b[idx];
            break;
        case Division:
            dest[idx] = a[idx] / b[idx];
    }
}

template <typename T>
__global__ void General_scalar_operation(const T* a, T scalar, T* dest, int n, Operations op)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= n)
        return;
    switch(op)
    {
        case Add:
            dest[idx] = a[idx] + scalar;
            break;
        case Subtract:
            dest[idx] = a[idx] - scalar;
            break;
        case Multiply:
            dest[idx] = a[idx] * scalar;
            break;
        case Division:
            dest[idx] = a[idx] / scalar;
    }
}



template <typename T>
void General_scalar_helper(const T* a, T scalar, T* dest, int n, Operations op)
{
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    T* cudaA, *cudaDest;

    cudaMalloc(&cudaA, n * sizeof(T));
    cudaMalloc(&cudaDest, n * sizeof(T));
    cudaMemcpy(cudaA, a, n * sizeof(T), cudaMemcpyHostToDevice);

    General_scalar_operation<T><<<blocks, threadsPerBlock>>>(cudaA, scalar, cudaDest, n, op);
    cudaDeviceSynchronize();

    cudaMemcpy(dest, cudaDest, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(cudaA);
    cudaFree(cudaDest);
}

template <typename T>
void General_operation_helper(const T* a, const T* b, T* dest, int n, Operations op)
{
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock -1) / threadsPerBlock;

    T* cudaA;
    T* cudaB;
    T* cudaDest;

    set_data(&cudaA, &cudaB, &cudaDest, n * sizeof(T));

    cudaMemcpy(cudaA, a, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, b, n * sizeof(T), cudaMemcpyHostToDevice);

    General_operation<T><<<blocks, threadsPerBlock>>>(cudaA, cudaB, cudaDest, n, op);
    cudaDeviceSynchronize();
    cudaMemcpy(dest, cudaDest, n * sizeof(T), cudaMemcpyDeviceToHost);
    free_data(cudaA, cudaB, cudaDest);
}

template <typename T>
__global__ void dot(const T* a, const T* b, T* dest, int m, int k, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if(idx < m && idy < n)
    {
        T sum = 0;
        for(int i = 0; i < k; i++)
            sum += a[idx * k + i] * b[i * n + idy];
        dest[idx * n + idy] = sum;
    }

}

template <typename T>
void dot_product(const T* a, const T* b, T* dest, int m, int k, int n)
{
    int threadX = 16;
    int blockX = (m + threadX - 1) / threadX;

    int threadY = 16;
    int blockY = (n + threadY - 1) / threadY;
 
    T* cudaA, *cudaB, *cudaDest;
    
    cudaMalloc(&cudaA, m * k * sizeof(T));
    cudaMalloc(&cudaB, n * k * sizeof(T));
    cudaMalloc(&cudaDest, m * n * sizeof(T));

    cudaMemcpy(cudaA, a, m * k * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, b, n * k * sizeof(T), cudaMemcpyHostToDevice);
        
    dim3 threads(threadX, threadY);
    dim3 Blocks(blockX, blockY);
    dot<<<Blocks, threads>>>(cudaA, cudaB, cudaDest, m, k, n);
    cudaDeviceSynchronize();

    cudaMemcpy(dest, cudaDest, m * n * sizeof(T), cudaMemcpyDeviceToHost);
    free_data(cudaA, cudaB, cudaDest);

}

template <typename T>
__global__ void sumting(T* a, int size, int iteration)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx > size || idx % (int)(pow(2,iteration+1)))
        return;
    a[idx] = a[idx] + (idx + pow(2, iteration) < size) * a[idx + (int)pow(2, iteration)];
}

template <typename T>
void sum_cuda(const T* a, T* dest, int size)
{
    T* cudaA;
    cudaMalloc(&cudaA, size * sizeof(T));
    cudaMemcpy(cudaA, a, size * sizeof(T), cudaMemcpyHostToDevice);

    int iterations = (int)(log2(size)) + 1;
    if(log2(size) == iterations-1)
        iterations--;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    for(int i = 0; i < iterations; i++)
    {
        sumting<T><<<blocks, threads>>>(cudaA, size, i);
        cudaDeviceSynchronize();
    }
    // sum_helper<<<1, 1>>>(cudaA, cudaDest, size, iterations);

    cudaMemcpy(dest, cudaA, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(cudaA);

}

template void General_operation_helper(const int* , const int*, int*, int, Operations);
template void General_operation_helper(const float* , const float*, float*, int, Operations);

template void General_scalar_helper(const int* a, int scalar, int* dest, int n, Operations);
template void General_scalar_helper(const float* a, float scalar, float* dest, int n, Operations);

template void dot_product(const int* , const int* , int*, int, int, int);
template void dot_product(const float* , const float* , float*, int, int, int);

template void sum_cuda(const int*, int*, int);
template void sum_cuda(const float*, float*, int);
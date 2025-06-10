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
__global__ void General_operation(const T* a, const T* b, T* dest, Dim2 a_dim, Dim2 b_dim, Operations op)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if(idx < a_dim.row && idy < a_dim.col)
    {
        int regular_index = idx * a_dim.col + idy; 
        bool row_equal = a_dim.row == b_dim.row;
        bool col_equal = a_dim.col == b_dim.col;
        bool regular_operation = row_equal && col_equal;

        int b_index = (regular_operation) * regular_index + (!regular_operation) * (row_equal * idx + col_equal * idy);
        switch(op)
        {
            case Add:
                dest[regular_index] = a[regular_index] + b[b_index];
                break;
            case Subtract:
                dest[regular_index] = a[regular_index] - b[b_index];
                break;
            case Multiply:
                dest[regular_index] = a[regular_index] * b[b_index];
                break;
            case Division:
                dest[regular_index] = a[regular_index] / b[b_index];
        }
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
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        std::cerr << "Cuda Scalar Error: " << cudaGetErrorString(error) << std::endl;

    cudaMemcpy(dest, cudaDest, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(cudaA);
    cudaFree(cudaDest);
}

template <typename T>
void General_operation_helper(const T* a, const T* b, T* dest, Operations op, Dim2 a_dim, Dim2 b_dim)
{
    int threadX = 16;
    int blockX = (a_dim.row + threadX - 1) / threadX;

    int threadY = 16;
    int blockY = (a_dim.col + threadY - 1) / threadY;


    T* cudaA;
    T* cudaB;
    T* cudaDest;

    cudaMalloc(&cudaA, a_dim.row * a_dim.col * sizeof(T));
    cudaMalloc(&cudaB, b_dim.row * b_dim.col * sizeof(T));
    cudaMalloc(&cudaDest, a_dim.row * a_dim.col * sizeof(T));

    cudaMemcpy(cudaA, a, a_dim.row * a_dim.col * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, b, b_dim.row * b_dim.col * sizeof(T), cudaMemcpyHostToDevice);

    dim3 threads(threadX, threadY);
    dim3 Blocks(blockX, blockY);
    General_operation<T><<<Blocks, threads>>>(cudaA, cudaB, cudaDest, a_dim, b_dim, op);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        std::cerr << "Cuda General Error: " << cudaGetErrorString(error) << std::endl;

    cudaMemcpy(dest, cudaDest, a_dim.row * a_dim.col * sizeof(T), cudaMemcpyDeviceToHost);
    free_data(cudaA, cudaB, cudaDest);
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
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        std::cerr << "Cuda Dot Error: " << cudaGetErrorString(error) << std::endl;

    cudaMemcpy(dest, cudaDest, m * n * sizeof(T), cudaMemcpyDeviceToHost);
    free_data(cudaA, cudaB, cudaDest);

}

template void General_operation_helper(const int* , const int*, int*, Operations, Dim2, Dim2);
template void General_operation_helper(const float* , const float*, float*, Operations, Dim2, Dim2);

template void General_scalar_helper(const int* a, int scalar, int* dest, int n, Operations);
template void General_scalar_helper(const float* a, float scalar, float* dest, int n, Operations);

template void dot_product(const int* , const int* , int*, int, int, int);
template void dot_product(const float* , const float* , float*, int, int, int);

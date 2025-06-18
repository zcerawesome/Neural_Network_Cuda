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
__global__ void num_equal_cuda(const T* a, const T* Y, int* numEqual, int size)
{
    __shared__ int blockcount;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= size)
        return;
    if(threadIdx.x == 0) blockcount = 0;
    __syncthreads();

    atomicAdd(&blockcount, (a[idx] == Y[idx]));
    __syncthreads();
    if(threadIdx.x == 0) atomicAdd(numEqual, blockcount);
}

template <typename T>
__global__ void largest_index_cuda(const T* a, T* Y, int numColumn, int numRow)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= numColumn * numRow)
        return;
    int index = 0;
    for(int i = 0; i < numRow; i++)
        index += (a[i * numColumn + idx] > a[index * numColumn + idx]) * (i - index);
    Y[idx] = (T)index;
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

    General_scalar_operation<<<blocks, threadsPerBlock>>>(a, scalar, dest, n, op);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        std::cerr << "Cuda Scalar Error: " << n << " "<< cudaGetErrorString(error) << std::endl;

}

template <typename T>
void General_operation_helper(const T* a, const T* b, T* dest, Operations op, Dim2 a_dim, Dim2 b_dim)
{
    int threadX = 16;
    int blockX = (a_dim.row + threadX - 1) / threadX;

    int threadY = 16;
    int blockY = (a_dim.col + threadY - 1) / threadY;


    dim3 threads(threadX, threadY);
    dim3 Blocks(blockX, blockY);
    General_operation<T><<<Blocks, threads>>>(a, b, dest, a_dim, b_dim, op);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cerr << "Cuda General Error: " << cudaGetErrorString(error) << std::endl;
        std::cerr << a_dim[0] << " " << a_dim[1] << "  " << b_dim[0] << " " << b_dim[1] << std::endl;
    }

}


template <typename T>
inline void dot_product(const T* a, const T* b, T* dest, int m, int k, int n)
{
    int threadX = 16;
    int blockX = (m + threadX - 1) / threadX;

    int threadY = 16;
    int blockY = (n + threadY - 1) / threadY;
 
    
            
    dim3 threads(threadX, threadY);
    dim3 Blocks(blockX, blockY);
    dot<<<Blocks, threads>>>(a, b, dest, m, k, n);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cerr << "Cuda Dot Error: " << cudaGetErrorString(error) << std::endl;
        std::cerr << m << " " << k << "  " << k << " " << n << std::endl;
    }

}

template <typename T>
__global__ void transpose_Cuda(const T* a, T* dest, Dim2 a_dim)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if(idx < a_dim.row && idy < a_dim.col)
        dest[idy * a_dim.row + idx] = a[idx * a_dim.col + idy];
}

template <typename T>
void transpose_GPU(const T* a, T* dest, Dim2 a_dim)
{
    int threadX = 16;
    int blockX = (a_dim.row + threadX - 1) / threadX;

    int threadY = 16;
    int blockY = (a_dim.col + threadY - 1) / threadY;
    dim3 threads(threadX, threadY);
    dim3 Blocks(blockX, blockY);

    transpose_Cuda<<<Blocks, threads>>>(a, dest, a_dim);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        std::cerr << "Cuda Transpose Error: " << cudaGetErrorString(error) << std::endl;
}

template <typename T>
void largest_index(const T* a, T* dest, Dim2 a_dim)
{
    int threads = 256;
    int blocks = (a_dim.col + threads - 1) / threads;
    largest_index_cuda<<<blocks, threads>>>(a, dest, a_dim.col, a_dim.row);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        std::cerr << "Cuda largest index Error: " << cudaGetErrorString(error) << std::endl;
}

template <typename T>
int num_equal(const T* a, const T* dest, int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    int* cudaSum;
    cudaMalloc(&cudaSum, sizeof(int));
    num_equal_cuda<<<blocks, threads>>>(a, dest, cudaSum, size);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        std::cerr << "Cuda num_equal Error: " << cudaGetErrorString(error) << std::endl;
    int num_equal1;
    cudaMemcpy(&num_equal1, cudaSum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(cudaSum);
    return num_equal1;   
}

template void General_operation_helper(const int* , const int*, int*, Operations, Dim2, Dim2);
template void General_operation_helper(const float* , const float*, float*, Operations, Dim2, Dim2);

template void General_scalar_helper(const int* a, int scalar, int* dest, int n, Operations);
template void General_scalar_helper(const float* a, float scalar, float* dest, int n, Operations);

template void dot_product(const int* , const int* , int*, int, int, int);
template void dot_product(const float* , const float* , float*, int, int, int);

template void transpose_GPU(const float* , float* , Dim2);
template void transpose_GPU(const int* , int* , Dim2);

template void largest_index(const float* , float* , Dim2);
template void largest_index(const int* , int* , Dim2);

template int num_equal(const float* , const float* , int);
template int num_equal(const int* , const int* , int);

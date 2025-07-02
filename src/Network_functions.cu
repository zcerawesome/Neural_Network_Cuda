#include "Network_functions.cuh"

Dim2 blocks_threads(int size, int threads)
{
    int thread_amount = threads;
    int blocks = (size + thread_amount - 1) / thread_amount;
    return {blocks, thread_amount};
}

__global__ void normal_sample_value(float* result, float* rand_floats, int size, float mu, float sigma)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= size)
        return;
    float z0 = sqrt(-2.0 * log(rand_floats[idx])) * cos(2.0 * M_PI * rand_floats[idx + size]);
    result[idx] = z0 * sigma + mu;
}

__global__ void generate_kernel(float* result, int size, unsigned long int seed)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if(id >= size)
        return;
    curandState state;
    curand_init(seed, id, 0, &state);
    result[id] = curand_uniform(&state) - 0.5;
}

__global__ void relu(const float* a, float* dest, int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < size)
        dest[idx] = a[idx] > 0 ? a[idx]: 0;
}

__global__ void relu_derive(const float* a, float* dest, int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < size)
        dest[idx] = a[idx] > 0;
}

__global__ void softmax(const float* a, float* dest, Dim2 a_dim)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < a_dim.col)
    {
        float sum = 0;
        for(int i = 0; i < a_dim.row; i++)
        {
            dest[i * a_dim.col + idx] = exp(a[i * a_dim.col + idx]);
            sum += dest[i * a_dim.col + idx];
        }
        for(int i = 0; i < a_dim.row; i++)
            dest[i * a_dim.col + idx] /= sum;
    }
}

__global__ void sigmoid(const float* a, float* dest, int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < size)
        dest[idx] = 1 / (1 + exp(-a[idx]));
}

__global__ void one_hot_encode(const float* a, float* dest, Dim2 a_dim)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < a_dim.col)
    {
        for(int i = 0; i < a_dim.row; i++)
        {
            dest[i * a_dim.col + idx] = 0;
        }
        dest[(int)(a[idx]) * a_dim.col + idx] = 1.0f;
    }
}

void randomize_matrix(matrice_gpu<float>& inp)
{
    int threads = 256;
    int blocks = (inp.size() + threads - 1) / threads;

    generate_kernel<<<blocks, threads>>>(inp.matrix, inp.size(), time(0));
    cudaDeviceSynchronize();
}

void random_sample(matrice_gpu<float>& inp, float mu, float sigma)
{
    Dim2 threads_block = blocks_threads(inp.size(), 256);
    matrice_gpu<float> rand_data(1, inp.size() * 2);
    randomize_matrix(rand_data);
    rand_data = rand_data + 0.5;
    normal_sample_value<<<threads_block[0], threads_block[1]>>>(inp.matrix, rand_data.matrix, inp.size(), mu, sigma);
    cudaDeviceSynchronize();
}

matrice_gpu<float> ReLU(matrice_gpu<float>& inp)
{
    matrice_gpu<float> temp(inp.numRows(), inp.numCols());
    int threads = 256;
    int blocks = (inp.size() + threads - 1) / threads;
    relu<<<blocks, threads>>>(inp.matrix, temp.matrix, inp.size());
    cudaDeviceSynchronize();
    return temp;
}

matrice_gpu<float> ReLU_derive(matrice_gpu<float>& inp)
{
    Dim2 threads_block = blocks_threads(inp.size(), 256);
    matrice_gpu<float> temp(inp.numRows(), inp.numCols());
    relu_derive<<<threads_block[0], threads_block[1]>>>(inp.matrix, temp.matrix, inp.size());
    cudaDeviceSynchronize();
    return temp;
}

matrice_gpu<float> softmax(matrice_gpu<float>& inp)
{
    Dim2 threads_block = blocks_threads(inp.numCols(), 256);
    matrice_gpu<float> temp(inp.numRows(), inp.numCols());
    softmax<<<threads_block[0], threads_block[1]>>>(inp.matrix, temp.matrix, {inp.numRows(), inp.numCols()});
    cudaDeviceSynchronize();
    return temp;
}

matrice_gpu<float> sigmoid(matrice_gpu<float>& inp)
{
    Dim2 threads_block = blocks_threads(inp.size(), 256);
    matrice_gpu<float> temp(inp.numRows(), inp.numCols());
    sigmoid<<<threads_block[0], threads_block[1]>>>(inp.matrix, temp.matrix, inp.size());
    cudaDeviceSynchronize();
    return temp;
}

matrice_gpu<float> one_hot_encode(matrice_gpu<float>& y, float max)
{
    matrice_gpu<float> temp(max + 1, y.numCols());
    Dim2 threads_block = blocks_threads(y.numCols(), 256);

    one_hot_encode<<<threads_block[0], threads_block[1]>>>(y.matrix, temp.matrix, {temp.numRows(), temp.numCols()});    
    cudaDeviceSynchronize();
    return temp;
}
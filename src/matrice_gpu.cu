#include "matrice_gpu.cuh"

template <typename T>
std::string matrice_gpu<T>::getShape()
{
    return std::to_string(row) + " " + std::to_string(col);
}

template <typename T>
Dim2 matrice_gpu<T>::shape()
{
    return {row, col};
}

template <typename T>
void matrice_gpu<T>::resize(int row, int col)
{
    this->row = row;
    this->col = col;
    if(matrix)
    {
        cudaFree(matrix);
    }
    cudaMalloc(&matrix, row * col * sizeof(T));
}

template <typename T>
void matrice_gpu<T>::toString()
{
    std::vector<T> data = CPU_data();
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
            std::cout << data[i * col + j] << " ";
        std::cout << std::endl;
    }
}

template <typename T>
int matrice_gpu<T>::numCols()
{
    return col;
}

template <typename T>
int matrice_gpu<T>::numRows()
{
    return row;
}

template <typename T>
int matrice_gpu<T>::size()
{
    return row * col;
}

template <typename T>
T matrice_gpu<T>::get(int x, int y)
{
    T data;
    cudaMemcpy(&data, matrix + x*col + y, sizeof(T), cudaMemcpyDeviceToHost);
    return data;
}

template <typename T>
std::vector<T> matrice_gpu<T>::CPU_data()
{
    std::vector<T> data(row * col);
    cudaMemcpy(data.data(), matrix, row * col * sizeof(T), cudaMemcpyDeviceToHost);
    return data;
}

template <typename T>
matrice_gpu<T>::matrice_gpu(){}

template <typename T>
matrice_gpu<T>::matrice_gpu(matrice_gpu<T>& inp)
{
    row = inp.row;
    col = inp.col;
    cudaMalloc(&matrix, row * col * sizeof(T));
    cudaMemcpy(matrix, inp.matrix, row * col * sizeof(T), cudaMemcpyDeviceToDevice);
}

template <typename T>
matrice_gpu<T>::matrice_gpu(const matrice_gpu<T>& inp)
{
    row = inp.row;
    col = inp.col;
    cudaMalloc(&matrix, row * col * sizeof(T));
    cudaMemcpy(matrix, inp.matrix, row * col * sizeof(T), cudaMemcpyDeviceToDevice);
}

template <typename T>
matrice_gpu<T>::matrice_gpu(int row, int col)
{
    this->row = row;
    this->col = col;
    cudaMalloc(&matrix, row * col * sizeof(T));
}

template <typename T>
matrice_gpu<T>::matrice_gpu(const std::vector<T>& inp)
{
    cudaMalloc(&matrix, inp.size() * sizeof(T));
    cudaMemcpy(matrix, inp.data(), inp.size() * sizeof(T), cudaMemcpyHostToDevice);
    row = 1;
    col = inp.size();
}

template <typename T>
matrice_gpu<T>::matrice_gpu(std::vector<T>& inp)
{
    cudaMalloc(&matrix, inp.size() * sizeof(T));
    cudaMemcpy(matrix, inp.data(), inp.size() * sizeof(T), cudaMemcpyHostToDevice);
    row = 1;
    col = inp.size();
}

template <typename T>
matrice_gpu<T>::matrice_gpu(const std::vector<std::vector<T>>& inp)
{
    row = inp.size();
    col = inp[0].size();
    cudaMalloc(&matrix, row * col * sizeof(T));
    for(int i = 0; i < row; i++)
        cudaMemcpy(matrix + i * col, inp[i].data(), col * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
matrice_gpu<T>::~matrice_gpu()
{
    if(matrix)
        cudaFree(matrix);
}

template <typename T>
matrice_gpu<T>::matrice_gpu(std::vector<std::vector<T>>& inp)
{
    row = inp.size();
    col = inp[0].size();
    cudaMalloc(&matrix, row * col * sizeof(T));
    for(int i = 0; i < row; i++)
        cudaMemcpy(matrix + i * col, inp[i].data(), col * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
matrice_gpu<T> matrice_gpu<T>::transpose()
{
    matrice_gpu<T> temp(col, row);
    transpose_GPU<T>(matrix, temp.matrix, {row, col});
    return temp;
}

template<typename T>
matrice_gpu<T> matrice_gpu<T>::operator=(const std::vector<std::vector<T>>& inp)
{
    return matrice_gpu<T>(inp);
}

template<typename T>
matrice_gpu<T> matrice_gpu<T>::operator=(std::vector<std::vector<T>>& inp)
{
    return matrice_gpu<T>(inp);
}

template<typename T>
matrice_gpu<T> matrice_gpu<T>::operator=(const std::vector<T>& inp)
{
    return matrice_gpu<T>(inp);
}

template<typename T>
matrice_gpu<T> matrice_gpu<T>::operator=(T inp)
{
    matrice_gpu<T> temp(1,1);
    cudaMemcpy(temp.matrix, &inp, sizeof(T), cudaMemcpyHostToDevice);
    return temp;
}

template<typename T>
matrice_gpu<T> matrice_gpu<T>::operator=(std::vector<T>& inp)
{
    return matrice_gpu<T>(inp);
}

template <typename T>
void matrice_gpu<T>::operator=(matrice_gpu<T>& inp)
{
    if(row != inp.row || col != inp.col || !matrix)
        resize(inp.row, inp.col);
    cudaMemcpy(matrix, inp.matrix, size() * sizeof(T), cudaMemcpyDeviceToDevice);
    row = inp.row;
    col = inp.col;
}

template <typename T>
void matrice_gpu<T>::operator=(const matrice_gpu<T>& inp)
{
    if(row != inp.row || col != inp.col || !matrix)
        resize(inp.row, inp.col);
    cudaMemcpy(matrix, inp.matrix, size() * sizeof(T), cudaMemcpyDeviceToDevice);
    row = inp.row;
    col = inp.col;
}

template <typename T>
void matrice_gpu<T>::update(int row, int col)
{
    this->row = row;
    this->col = col;
}

template <typename T>
T& matrice_gpu<T>::operator[](int value)
{ 
    return matrix[value];
}

template <typename T>
T* matrice_gpu<T>::getData()
{
    return matrix;
}

template <typename T>
T matrice_gpu<T>::sum()
{
    T value = 0;
    std::vector<T> cpu_data = CPU_data();
    for(auto& val: cpu_data)
        value += val;
    return value;
}

template <typename T>
T matrice_gpu<T>::max()
{
    std::vector<T> cpu_data = CPU_data();
    T maximum = cpu_data[0];
    for(T& val: cpu_data)
        maximum = val > maximum ? val: maximum;
    return maximum;
}

template <typename T>
int matrice_gpu<T>::largest_index(int col)
{
    std::vector<T> cpu_data = CPU_data();
    int index = 0;
    for(int i = 0; i < row; i++)
        index = cpu_data[i * this->col + col] > cpu_data[i * this->col + col] ? i: index;
    return index;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::getRows(int start, int end)
{
    matrice_gpu<T> temp(end - start, col);
    cudaMemcpy(temp.matrix, matrix + start * col, (end - start) * col * sizeof(T), cudaMemcpyDeviceToDevice);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::getCols(int start, int end)
{
    matrice_gpu<T> temp = transpose();
    temp = temp.getRows(start, end);
    temp = temp.transpose();
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator-(matrice_gpu<T>& inp)
{
    matrice_gpu<T> temp(inp.row, inp.col);
    General_operation_helper<T>(matrix, inp.getData(), temp.getData(), Subtract, {row, col}, {inp.row, inp.col});
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator-(const matrice_gpu<T>& inp)
{
    matrice_gpu<T> temp(inp.row, inp.col);
    matrice_gpu<T> inp2 = inp;
    General_operation_helper<T>(matrix, inp2.getData(), temp.getData(), Subtract, {row, col}, {inp.row, inp.col});
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator-(T inp)
{
    matrice_gpu<T> temp(row, col);
    General_scalar_helper(matrix, inp, temp.getData(), row * col, Subtract);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator+(matrice_gpu<T>& inp)
{
    matrice_gpu<T> temp(row, col);
    General_operation_helper<T>(matrix, inp.getData(), temp.getData(), Add, {row, col}, {inp.row, inp.col});
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator+(const matrice_gpu<T>& inp)
{
    matrice_gpu<T> temp(row, col);
    matrice_gpu<T> inp2 = inp;
    General_operation_helper<T>(matrix, inp2.getData(), temp.getData(), Add, {row, col}, {inp.row, inp.col});
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator+(T inp)
{
    matrice_gpu<T> temp(row, col);
    General_scalar_helper(matrix, inp, temp.getData(), row * col, Add);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator*(matrice_gpu<T>& inp)
{
    if(row != inp.row || col != inp.col)
        std::cerr << "Error different dimensions in multiplication" << std::endl;
    matrice_gpu<T> temp(inp.row, inp.col);
    General_operation_helper<T>(matrix, inp.getData(), temp.getData(), Multiply, {row, col}, {inp.row, inp.col});

    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator*(const matrice_gpu<T>& inp)
{
    if(row != inp.row || col != inp.col)
        std::cerr << "Error different dimensions in multiplication" << std::endl;
    matrice_gpu<T> temp(inp.row, inp.col);
    matrice_gpu<T> inp2 = inp;
    General_operation_helper<T>(matrix, inp2.getData(), temp.getData(), Multiply, {row, col}, {inp.row, inp.col});
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator*(T inp)
{

    matrice_gpu<T> temp(row, col);
    General_scalar_helper(matrix, inp, temp.getData(), row * col, Multiply);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator/(T inp)
{
    matrice_gpu<T> temp(row, col);
    General_scalar_helper(matrix, inp, temp.matrix, size(), Division);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::Dot(const matrice_gpu<T>& inp)
{
    if(col != inp.row)
    {
        std::cerr << "Error wrong dimensions for dot product " << row << " " << col << std::endl;
        std::cerr << inp.row << " " << inp.col << std::endl;
        exit(0);
        return {};
    }
    matrice_gpu<T> temp(row, inp.col);
    matrice_gpu<T> inp2 = inp;
    dot_product(getData(), inp2.getData(), temp.getData(), row, inp.row, inp.col);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::Dot(matrice_gpu<T>& inp)
{
    if(col != inp.row)
    {
        std::cerr << "Error wrong dimensions for dot product " << row << " " << col << std::endl;
        std::cerr << inp.row << " " << inp.col << std::endl;
        exit(0);
        return {};
    }
    matrice_gpu<T> temp(row, inp.col);
    dot_product(getData(), inp.getData(), temp.getData(), row, inp.row, inp.col);
    return temp;
}

template <typename T>
T* matrice_gpu<T>::begin()
{
    return matrix;
}

template <typename T>
T* matrice_gpu<T>::end()
{
    return matrix + size();
}

template <typename T>
const T* matrice_gpu<T>::begin() const
{
    return matrix;
}

template <typename T>
const T* matrice_gpu<T>::end() const
{
    return matrix + size();
}
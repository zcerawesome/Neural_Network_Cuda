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
    if(matrix)
        cudaFree(matrix);
    cudaMalloc(&matrix, row * col * sizeof(T));
    this->row = row;
    this->col = col;
}

template <typename T>
void matrice_gpu<T>::update(int row, int col)
{
    this->row = row;
    this->col = col;
}

template <typename T>
std::string matrice_gpu<T>::toString()
{
    if(size() == 0)
        return "";
    std::vector<T> cpu = CPU_data();
    std::string data;
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            data += std::to_string(cpu[i * col + j]);
            data += (j == col -1) ? ' ': ',';
        }
        data += '\n';
    }
    return data;
}

template <typename T>
int matrice_gpu<T>::numRows()
{
    return row;
}

template <typename T>
int matrice_gpu<T>::numCols()
{
    return col;
}

template <typename T>
T matrice_gpu<T>::get(int x, int y)
{
    T value;
    cudaMemcpy(&value, matrix + x * col + y, sizeof(T), cudaMemcpyDeviceToHost);
    return value;
}

template <typename T>
std::vector<T> matrice_gpu<T>::CPU_data()
{
    std::vector<T> cpu(row * col);
    cudaMemcpy(cpu.data(), matrix, row * col * sizeof(T), cudaMemcpyDeviceToHost);
    return cpu;
}

template <typename T>
int matrice_gpu<T>::size()
{
    return row * col;
}

template <typename T>
void matrice_gpu<T>::deletePointer() const
{
    void** matrix2 = (void**)(&matrix);
    *matrix2 = 0;
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
    matrix = inp.matrix;
    inp.deletePointer();
}

template <typename T>
matrice_gpu<T>::matrice_gpu(int row, int col): row(row), col(col)
{
    cudaMalloc(&matrix, row * col * sizeof(T));
}

template <typename T>
matrice_gpu<T>::matrice_gpu(std::vector<T>& inp)
{
    row = 1;
    col = inp.size();
    cudaMalloc(&matrix, row * col * sizeof(T));
    cudaMemcpy(matrix, inp.data(), row * col * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
matrice_gpu<T>::matrice_gpu(const std::vector<T>& inp)
{
    row = 1;
    col = inp.size();
    cudaMalloc(&matrix, row * col * sizeof(T));
    cudaMemcpy(matrix, inp.data(), row * col * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
matrice_gpu<T>::matrice_gpu(std::vector<std::vector<T>>& inp): row(inp.size()), col(inp[0].size())
{
    cudaMalloc(&matrix, row * col * sizeof(T));
    for(int i = 0; i < row; i++)
        cudaMemcpy(matrix + i * col, inp[i].data(), col * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
matrice_gpu<T>::matrice_gpu(const std::vector<std::vector<T>>& inp): row(inp.size()), col(inp[0].size())
{
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
matrice_gpu<T> matrice_gpu<T>::transpose()
{
    if(row == 1 || col == 1)
    {
        matrice_gpu<T> temp = *this;
        temp.update(col, row);
        return temp;
    }
    matrice_gpu<T> temp(col, row);
    transpose_GPU(matrix, temp.matrix, {row, col});
    return temp;
}

template <typename T>
void matrice_gpu<T>::operator=(std::vector<std::vector<T>>& inp)
{
    row = inp.size();
    col = inp[0].size();
    cudaMalloc(&matrix, row * col * sizeof(T));
    for(int i = 0; i < row; i++)
        cudaMemcpy(matrix + i * col, inp[i].data(), col * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void matrice_gpu<T>::operator=(const std::vector<std::vector<T>>& inp)
{
    row = inp.size();
    col = inp[0].size();
    cudaMalloc(&matrix, row * col * sizeof(T));
    for(int i = 0; i < row; i++)
        cudaMemcpy(matrix + i * col, inp[i].data(), col * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void matrice_gpu<T>::operator=(std::vector<T>& inp)
{
    row = 1;
    col = inp.size();
    cudaMalloc(&matrix, row * col * sizeof(T));
    cudaMemcpy(matrix, inp.data(), row * col * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void matrice_gpu<T>::operator=(const std::vector<T>& inp)
{
    row = 1;
    col = inp.size();
    cudaMalloc(&matrix, row * col * sizeof(T));
    cudaMemcpy(matrix, inp.data(), row * col * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void matrice_gpu<T>::operator=(T inp)
{
    row = 1;
    col = 1;
    cudaMalloc(&matrix, sizeof(T));
    cudaMemcpy(matrix, &inp, sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void matrice_gpu<T>::operator=(matrice_gpu<T>& inp)
{
    row = inp.row;
    col = inp.col;
    if(matrix)
        cudaFree(matrix);
    cudaMalloc(&matrix, row * col * sizeof(T));
    cudaMemcpy(matrix, inp.matrix, row * col * sizeof(T), cudaMemcpyDeviceToDevice);
}

template <typename T>
void matrice_gpu<T>::operator=(const matrice_gpu<T>& inp)
{
    row = inp.row;
    col = inp.col;
    matrix = inp.matrix;
    inp.deletePointer();
}

template <typename T>
void matrice_gpu<T>::load_data(std::string file_name, bool header, int maxRows, int row_number)
{
    std::vector<std::vector<T>> data;
    std::ifstream file(file_name);
    if(!file.is_open())
    {
        std::cerr << "Error file cannot be opened" << std::endl;
        return;
    }
    std::string line;
    if(header)
        getline(file, line);
    int i = 0;
    while(i < row_number && getline(file, line))
        i++;
    i = 0;
    while(getline(file, line) && (maxRows == -1 || i < maxRows))
    {
        std::vector<T> row;
        const char* ptr = line.c_str();
        char* end;

        while(*ptr)
        {
            T val = std::strtof(ptr, &end);
            row.push_back(val);
            if(end == ptr)
                ++ptr;
            else
                ptr = end + 1;
        }
        data.push_back(std::move(row));
        i++;
    }
    *this = data;
}

template <typename T>
T& matrice_gpu<T>::operator[](int row)
{
    return matrix[row];
}

template <typename T>
T matrice_gpu<T>::sum()
{
    std::vector<T> cpu = CPU_data();
    T val = 0;
    for(auto value: cpu)
        val += value;
    return val;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::sum_axis(int axis)
{
    int rows = 1;
    int cols = 1;

    if(axis)
        rows = row;
    else
        cols = col;
    std::vector<T> temp_vec(rows * cols);
    std::vector<T> cpu = CPU_data();
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            int index = (rows > cols) ? i: j;
            temp_vec[index] += cpu[i * numCols() + j];
        }
    }
    matrice_gpu<T> temp(temp_vec);
    if(rows != 1)
        temp = temp.transpose();
    return temp;
}

template <typename T>
T matrice_gpu<T>::max()
{
    std::vector<T> cpu = CPU_data();
    T val = cpu[0];
    for(auto value: cpu)
        val = value > val? value: val;
    return val;
}

template <typename T>
int matrice_gpu<T>::largest_index(int col)
{
    std::vector<T> cpu_data = CPU_data();
    int largest_index = 0;
    for(int i = 0; i < row; i++)
        largest_index = cpu_data[i * this->col + col] > cpu_data[largest_index * this->col + col] ? i: largest_index;
    return largest_index;
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
matrice_gpu<T> matrice_gpu<T>::clamp(T lower, T higher)
{
    matrice_gpu<T> temp(row, col);
    general_clamp(matrix, temp.matrix, size(), lower, higher);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator-(matrice_gpu<T>& inp)
{
    matrice_gpu<T> temp(row, col);
    General_operation_helper(matrix, inp.matrix, temp.matrix, Subtract, {row, col}, {inp.row, inp.col});
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator-(const matrice_gpu<T>& inp)
{
    matrice_gpu<T> temp(row, col);
    General_operation_helper(matrix, inp.matrix, temp.matrix, Subtract, {row, col}, {inp.row, inp.col});
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator-(T inp)
{
    matrice_gpu<T> temp(row, col);
    General_scalar_helper(matrix, inp, temp.matrix, row * col, Subtract);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator+(matrice_gpu<T>& inp)
{
    matrice_gpu<T> temp(row, col);
    General_operation_helper(matrix, inp.matrix, temp.matrix, Add, {row, col}, {inp.row, inp.col});
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator+(const matrice_gpu<T>& inp)
{
    matrice_gpu<T> temp(row, col);
    General_operation_helper(matrix, inp.matrix, temp.matrix, Add, {row, col}, {inp.row, inp.col});
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator+(T inp)
{
    matrice_gpu<T> temp(row, col);
    General_scalar_helper(matrix, inp, temp.matrix, row * col, Add);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator*(matrice_gpu<T>& inp)
{
    matrice_gpu<T> temp(row, col);
    General_operation_helper(matrix, inp.matrix, temp.matrix, Multiply, {row, col}, {inp.row, inp.col});
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator*(const matrice_gpu<T>& inp)
{
    matrice_gpu<T> temp(row, col);
    General_operation_helper(matrix, inp.matrix, temp.matrix, Multiply, {row, col}, {inp.row, inp.col});
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator*(T inp)
{
    matrice_gpu<T> temp(row, col);
    General_scalar_helper(matrix, inp, temp.matrix, row * col, Multiply);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator/(T inp)
{
    matrice_gpu<T> temp(row, col);
    General_scalar_helper(matrix, inp, temp.matrix, row * col, Division);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::Dot(const matrice_gpu<T>& inp)
{
    matrice_gpu<T> temp(row, inp.col);
    dot_product(matrix, inp.matrix, temp.matrix, row, col, inp.col);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::Dot(matrice_gpu<T>& inp)
{
    matrice_gpu<T> temp(row, inp.col);
    dot_product(matrix, inp.matrix, temp.matrix, row, col, inp.col);
    return temp;
}

template <typename T>
int num_correct(matrice_gpu<T>& output, matrice_gpu<T>& answer)
{
    matrice_gpu<T> vals_correct(1, answer.numCols());
    largest_index(output.matrix, vals_correct.matrix, {output.numRows(), output.numCols()});
    std::vector<T> answers = answer.CPU_data();
    std::vector<T> guess = vals_correct.CPU_data();
    int correct = 0;
    for(int i = 0; i < answer.numCols(); i++)
        if(guess[i] == answers[i])
            correct++;
    return correct;
}
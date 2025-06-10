#include "matrice_gpu.h"

template <typename T>
std::vector<int> matrice_gpu<T>::shape()
{
    return {row, col};
}

template <typename T>
void matrice_gpu<T>::resize(int row, int col)
{
    this->row = row;
    this->col = col;
    matrix.resize(row * col);
}

template <typename T>
void matrice_gpu<T>::toString()
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
            std::cout << matrix[i * col + j] << " ";
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
T& matrice_gpu<T>::get(int x, int y)
{
    return matrix[x * col + y];
}

template <typename T>
matrice_gpu<T>::matrice_gpu(){}

template <typename T>
matrice_gpu<T>::matrice_gpu(int row, int col)
{
    this->row = row;
    this->col = col;
    matrix.resize(row * col);
}

template <typename T>
matrice_gpu<T>::matrice_gpu(const std::vector<T>& inp)
{
    matrix = inp;
    row = 1;
    col = inp.size();
}

template <typename T>
matrice_gpu<T>::matrice_gpu(std::vector<T>& inp)
{
    matrix = inp;
    row = 1;
    col = inp.size();
}

template <typename T>
matrice_gpu<T>::matrice_gpu(const std::vector<std::vector<T>>& inp)
{
    row = inp.size();
    col = inp[0].size();
    matrix.resize(row * col);
    for(int i = 0; i < row; i++)
        for(int j = 0; j < col; j++)
            matrix[i * col + j] = inp[i][j];
}

template <typename T>
matrice_gpu<T>::matrice_gpu(std::vector<std::vector<T>>& inp)
{
    row = inp.size();
    col = inp[0].size();
    matrix.resize(row * col);
    for(int i = 0; i < row; i++)
        for(int j = 0; j < col; j++)
            matrix[i * col + j] = inp[i][j];
}

template<typename T>
matrice_gpu<T> matrice_gpu<T>::transpose()
{
    matrice_gpu<T> temp(col, row);
    for(int j = 0; j < row; j++)
        for(int i = 0; i < col; i++)
            temp.get(i, j) = get(j, i);
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
matrice_gpu<T> matrice_gpu<T>::operator=(std::vector<T>& inp)
{
    return matrice_gpu<T>(inp);
}

template <typename T>
void matrice_gpu<T>::operator=(matrice_gpu<T>& inp)
{
    matrix = inp.matrix;
    row = inp.row;
    col = inp.col;
}

template <typename T>
void matrice_gpu<T>::operator=(const matrice_gpu<T>& inp)
{
    matrix = inp.matrix;
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
    return matrix.data();
}

template <typename T>
T matrice_gpu<T>::sum()
{
    T value = 0;
    for(int i = 0; i < row * col; i++)
        value += matrix[i];
    return value;
}

template <typename T>
T matrice_gpu<T>::max()
{
    T maximum = matrix[0];
    for(T& val: matrix)
        maximum = val > maximum ? val: maximum;
    return maximum;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::getRows(int start, int end)
{
    matrice_gpu<T> temp(end - start, col);
    for(int i = start; i < end; i++)
        for(int j = 0; j < col; j++)
            temp.get(i-start,j) = get(i, j);
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
    if(inp.row != row || inp.col != col || matrix.size() < 1000)
        general_operation(*this, inp, temp, Subtract);
    else
        General_operation_helper<T>(matrix.data(), inp.getData(), temp.getData(), Subtract, {row, col}, {inp.row, inp.col});
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator-(const matrice_gpu<T>& inp)
{
    matrice_gpu<T> temp(inp.row, inp.col);
    matrice_gpu<T> inp2 = inp;
    if(inp.row != row || inp.col != col || matrix.size() < 1000)
        general_operation(*this, inp2, temp, Subtract);
    else
        General_operation_helper<T>(matrix.data(), inp2.getData(), temp.getData(), Subtract, {row, col}, {inp.row, inp.col});
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator-(T inp)
{
    matrice_gpu<T> temp(row, col);
    if(matrix.size() > 1000)
        General_scalar_helper(matrix.data(), inp, temp.getData(), row * col, Subtract);
    else
        general_Scalar_operation(*this, inp, temp, Subtract);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator+(matrice_gpu<T>& inp)
{
    matrice_gpu<T> temp(row, col);
    if(row != inp.row || col != inp.col || matrix.size() < 1000)
        general_operation(*this, inp, temp, Add);
    else
        General_operation_helper<T>(matrix.data(), inp.getData(), temp.getData(), Add, {row, col}, {inp.row, inp.col});
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator+(const matrice_gpu<T>& inp)
{
    matrice_gpu<T> temp(row, col);
    matrice_gpu<T> inp2 = inp;
    if(inp.row != row || inp.col != col || matrix.size() < 1000)
        general_operation(*this, inp2, temp, Add);
    else 
        General_operation_helper<T>(matrix.data(), inp2.getData(), temp.getData(), Add, {row, col}, {inp.row, inp.col});
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator+(T inp)
{
    matrice_gpu<T> temp(row, col);
    if(matrix.size() > 1000)
        General_scalar_helper(matrix.data(), inp, temp.getData(), row * col, Add);
    else
        general_Scalar_operation(*this, inp, temp, Add);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator*(matrice_gpu<T>& inp)
{
    if(row != inp.row || col != inp.col)
        std::cerr << "Error different dimensions in multiplication" << std::endl;
    matrice_gpu<T> temp(inp.row, inp.col);
    if(matrix.size() > 1000)
        General_operation_helper<T>(matrix.data(), inp.getData(), temp.getData(), Multiply, {row, col}, {inp.row, inp.col});
    else
        general_operation(*this, inp, temp, Multiply);

    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator*(const matrice_gpu<T>& inp)
{
    if(row != inp.row || col != inp.col)
        std::cerr << "Error different dimensions in multiplication" << std::endl;
    matrice_gpu<T> temp(inp.row, inp.col);
    matrice_gpu<T> inp2 = inp;
    if(matrix.size() > 1000)
        General_operation_helper<T>(matrix.data(), inp2.getData(), temp.getData(), Multiply, {row, col}, {inp.row, inp.col});
    else
        general_operation(*this, inp2, temp, Multiply);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator*(T inp)
{

    matrice_gpu<T> temp(row, col);
    if(matrix.size() > 1000)
        General_scalar_helper(matrix.data(), inp, temp.getData(), row * col, Multiply);
    else
        general_Scalar_operation(*this, inp, temp, Multiply);
    return temp;
}

template <typename T>
matrice_gpu<T> matrice_gpu<T>::operator/(T inp)
{
    matrice_gpu<T> temp(row, col);
    if(matrix.size() > 1000)
        General_scalar_helper(matrix.data(), inp, temp.getData(), row * col, Division);
    else
        general_Scalar_operation(*this, inp, temp, Division);
    return temp;
}

template <typename T>
inline void matrice_gpu<T>::dot(matrice_gpu<T>& a, matrice_gpu<T>& b, matrice_gpu<T>& dest)
{
    for(int i = 0; i < a.row; i++)
    {
        for(int j = 0; j < b.col; j++)
        {
            T& value = dest.get(i, j);
            value = 0;
            for(int k = 0; k < a.col; k++)
                value += a.get(i, k) * b.get(k, j);
        }
    }
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
    if(matrix.size() > 1000)
        dot_product(getData(), inp2.getData(), temp.getData(), row, inp.row, inp.col);
    else
        dot(*this, inp2, temp);
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
    if(matrix.size() > 1000)
        dot_product(getData(), inp.getData(), temp.getData(), row, inp.row, inp.col);
    else
        dot(*this, inp, temp);
        // std::cout << "CPU dot" << std::endl;
    return temp;
}

template <typename T>
void general_Scalar_operation(matrice_gpu<T>& a, T b, matrice_gpu<T>& dest, Operations ops)
{
    for(int i = 0; i < a.numRows(); i++)
    {
        for(int j = 0; j < a.numCols(); j++)
        {
            int index = i * dest.numCols() + j;
            switch(ops)
            {
                case Add:
                    dest[index] = a[index] + b;
                    break;
                case Subtract:
                    dest[index] = a[index] - b;
                    break;
                case Multiply:
                    dest[index] = a[index] * b;
                    break;
                case Division:
                    dest[index] = a[index] / b;
            }
        }
    }
}

template <typename T>
void general_operation(matrice_gpu<T>& a, matrice_gpu<T>& b, matrice_gpu<T>& dest, Operations ops)
{
    bool vector_operation[] = {a.numRows() != b.numRows(), a.numCols() != b.numCols()};
    for(int i = 0; i < a.numRows(); i++)
    {
        for(int j = 0; j < a.numCols(); j++)
        {
            int index = i * dest.numCols() + j;
            int b_index = index;
            if(vector_operation[0] || vector_operation[1])
                b_index = vector_operation[0] ? j:i;
            switch(ops)
            {
                case Add:
                    dest[index] = a[index] + b[b_index];
                    break;
                case Subtract:
                    dest[index] = a[index] - b[b_index];
                    break;
                case Multiply:
                    dest[index] = a[index] * b[b_index];
                    break;
                case Division:
                    dest[index] = a[index] / b[b_index];
            }
        }
    }
}
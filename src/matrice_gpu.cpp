#include "matrice_gpu.h"

template <typename T>
std::vector<int> matrice<T>::shape()
{
    return {row, col};
}

template <typename T>
void matrice<T>::resize(int row, int col)
{
    this->row = row;
    this->col = col;
    matrix.resize(row * col);
}

template <typename T>
void matrice<T>::toString()
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
            std::cout << matrix[i * col + j] << " ";
        std::cout << std::endl;
    }
}

template <typename T>
int matrice<T>::numCols()
{
    return col;
}

template <typename T>
int matrice<T>::numRows()
{
    return row;
}

template <typename T>
T& matrice<T>::get(int x, int y)
{
    return matrix[x * col + y];
}

template <typename T>
matrice<T>::matrice(){}

template <typename T>
matrice<T>::matrice(int row, int col)
{
    this->row = row;
    this->col = col;
    matrix.resize(row * col);
}

template <typename T>
matrice<T>::matrice(const std::vector<T>& inp)
{
    matrix = inp;
    row = 1;
    col = inp.size();
}

template <typename T>
matrice<T>::matrice(std::vector<T>& inp)
{
    matrix = inp;
    row = 1;
    col = inp.size();
}

template <typename T>
matrice<T>::matrice(const std::vector<std::vector<T>>& inp)
{
    row = inp.size();
    col = inp[0].size();
    matrix.resize(row * col);
    for(int i = 0; i < row; i++)
        for(int j = 0; j < col; j++)
            matrix[i * col + j] = inp[i][j];
}

template <typename T>
matrice<T>::matrice(std::vector<std::vector<T>>& inp)
{
    row = inp.size();
    col = inp[0].size();
    matrix.resize(row * col);
    for(int i = 0; i < row; i++)
        for(int j = 0; j < col; j++)
            matrix[i * col + j] = inp[i][j];
}

template<typename T>
matrice<T> matrice<T>::transpose()
{
    matrice<T> temp(col, row);
    for(int j = 0; j < row; j++)
        for(int i = 0; i < col; i++)
            temp.get(i, j) = get(j, i);
    return temp;
}

template<typename T>
matrice<T> matrice<T>::operator=(const std::vector<std::vector<T>>& inp)
{
    return matrice<T>(inp);
}

template<typename T>
matrice<T> matrice<T>::operator=(std::vector<std::vector<T>>& inp)
{
    return matrice<T>(inp);
}

template<typename T>
matrice<T> matrice<T>::operator=(const std::vector<T>& inp)
{
    return matrice<T>(inp);
}

template<typename T>
matrice<T> matrice<T>::operator=(std::vector<T>& inp)
{
    return matrice<T>(inp);
}

template <typename T>
void matrice<T>::operator=(matrice<T>& inp)
{
    matrix = inp.matrix;
    row = inp.row;
    col = inp.col;
}

template <typename T>
void matrice<T>::operator=(const matrice<T>& inp)
{
    matrix = inp.matrix;
    row = inp.row;
    col = inp.col;
}

template <typename T>
void matrice<T>::update(int row, int col)
{
    this->row = row;
    this->col = col;
}

template <typename T>
T& matrice<T>::operator[](int value)
{
    return matrix[value];
}

template <typename T>
T* matrice<T>::getData()
{
    return matrix.data();
}

template <typename T>
T matrice<T>::sum()
{
    T value = 0;
    for(int i = 0; i < row * col; i++)
        value += matrix[i];
    return value;
}

template <typename T>
T matrice<T>::max()
{
    T maximum = matrix[0];
    for(T& val: matrix)
        maximum = val > maximum ? val: maximum;
    return maximum;
}

template <typename T>
matrice<T> matrice<T>::getRows(int start, int end)
{
    matrice<T> temp(end - start, col);
    for(int i = start; i < end; i++)
        for(int j = 0; j < col; j++)
            temp.get(i-start,j) = get(i, j);
    return temp;
}

template <typename T>
matrice<T> matrice<T>::getCols(int start, int end)
{
    matrice<T> temp = transpose();
    temp = temp.getRows(start, end);
    temp = temp.transpose();
    return temp;
}

template <typename T>
matrice<T> matrice<T>::operator-(matrice<T>& inp)
{
    matrice<T> temp(inp.row, inp.col);
    if(inp.row != row)
        general_vector_operation(*this, inp, temp, Subtract, 0);
    else if(inp.col != col)
        general_vector_operation(*this, inp, temp, Subtract, 1);
    if(matrix.size() > 1000)
        General_operation_helper<T>(matrix.data(), inp.getData(), temp.getData(), inp.row * inp.col, Subtract);
    else
        general_operation(*this, inp, temp, Subtract);
    return temp;
}

template <typename T>
matrice<T> matrice<T>::operator-(const matrice<T>& inp)
{
    matrice<T> temp(inp.row, inp.col);
    matrice<T> inp2 = inp;
    if(inp.row != row)
        general_vector_operation(*this, inp2, temp, Subtract, 0);
    else if(inp.col != col)
        general_vector_operation(*this, inp2, temp, Subtract, 1);
    if(matrix.size() > 1000)
        General_operation_helper<T>(matrix.data(), inp2.getData(), temp.getData(), inp.row * inp.col, Subtract);
    else
        general_operation(*this, inp2, temp, Subtract);
    return temp;
}

template <typename T>
matrice<T> matrice<T>::operator-(T inp)
{
    matrice<T> temp(row, col);
    if(matrix.size() > 1000)
        General_scalar_helper(matrix.data(), inp, temp.getData(), row * col, Subtract);
    else
        general_Scalar_operation(*this, inp, temp, Subtract);
    return temp;
}

template <typename T>
matrice<T> matrice<T>::operator+(matrice<T>& inp)
{
    matrice<T> temp(row, col);
    if(inp.row != row)
        general_vector_operation(*this, inp, temp, Add, 0);
    else if(inp.col != col)
        general_vector_operation(*this, inp, temp, Add, 1);
    if(matrix.size() > 1000)
        General_operation_helper<T>(matrix.data(), inp.getData(), temp.getData(), inp.row * inp.col, Add);
    else
        general_operation(*this, inp, temp, Add);
    return temp;
}

template <typename T>
matrice<T> matrice<T>::operator+(const matrice<T>& inp)
{
    matrice<T> temp(row, col);
    matrice<T> inp2 = inp;
    if(inp.row != row)
        general_vector_operation(*this, inp2, temp, Add, 0);
    else if(inp.col != col)
        general_vector_operation(*this, inp2, temp, Add, 1);
    if(matrix.size() > 1000)
        General_operation_helper<T>(matrix.data(), inp2.getData(), temp.getData(), inp.row * inp.col, Add);
    else
        general_operation(*this, inp2, temp, Add);
    return temp;
}

template <typename T>
matrice<T> matrice<T>::operator+(T inp)
{
    matrice<T> temp(row, col);
    if(matrix.size() > 1000)
        General_scalar_helper(matrix.data(), inp, temp.getData(), row * col, Add);
    else
        general_Scalar_operation(*this, inp, temp, Add);
    return temp;
}

template <typename T>
matrice<T> matrice<T>::operator*(matrice<T>& inp)
{
    if(row != inp.row || col != inp.col)
        std::cerr << "Error different dimensions in multiplication" << std::endl;
    matrice<T> temp(inp.row, inp.col);
    if(matrix.size() > 1000)
        General_operation_helper<T>(matrix.data(), inp.getData(), temp.getData(), inp.row * inp.col, Multiply);
    else
        general_operation(*this, inp, temp, Multiply);

    return temp;
}

template <typename T>
matrice<T> matrice<T>::operator*(const matrice<T>& inp)
{
    if(row != inp.row || col != inp.col)
        std::cerr << "Error different dimensions in multiplication" << std::endl;
    matrice<T> temp(inp.row, inp.col);
    matrice<T> inp2 = inp;
    if(matrix.size() > 1000)
        General_operation_helper<T>(matrix.data(), inp2.getData(), temp.getData(), inp.row * inp.col, Multiply);
    else
        general_operation(*this, inp2, temp, Multiply);
    return temp;
}

template <typename T>
matrice<T> matrice<T>::operator*(T inp)
{

    matrice<T> temp(row, col);
    if(matrix.size() > 1000)
        General_scalar_helper(matrix.data(), inp, temp.getData(), row * col, Multiply);
    else
        general_Scalar_operation(*this, inp, temp, Multiply);
    return temp;
}

template <typename T>
matrice<T> matrice<T>::operator/(T inp)
{
    matrice<T> temp(row, col);
    if(matrix.size() > 1000)
        General_scalar_helper(matrix.data(), inp, temp.getData(), row * col, Division);
    else
        general_Scalar_operation(*this, inp, temp, Division);
    return temp;
}

template <typename T>
inline void matrice<T>::dot(matrice<T>& a, matrice<T>& b, matrice<T>& dest)
{
    for(int i = 0; i < a.row; i++)
    {
        for(int j = 0; j < b.col; j++)
        {
            T& value = dest[i * dest.col + j];
            value = 0;
            for(int k = 0; k < a.col; k++)
                value += a[i * a.col + k] * b[k * b.col + j];
        }
    }
}

template <typename T>
matrice<T> matrice<T>::Dot(const matrice<T>& inp)
{
    if(col != inp.row)
    {
        std::cerr << "Error wrong dimensions for dot product " << row << " " << col << std::endl;
        std::cerr << inp.row << " " << inp.col << std::endl;
        exit(0);
        return {};
    }
    matrice<T> temp(row, inp.col);
    matrice<T> inp2 = inp;
    if(matrix.size() > 1000)
        dot_product(getData(), inp2.getData(), temp.getData(), row, inp.row, inp.col);
    else
        dot(*this, inp2, temp);
    return temp;
}

template <typename T>
matrice<T> matrice<T>::Dot(matrice<T>& inp)
{
    if(col != inp.row)
    {
        std::cerr << "Error wrong dimensions for dot product " << row << " " << col << std::endl;
        std::cerr << inp.row << " " << inp.col << std::endl;
        exit(0);
        return {};
    }
    matrice<T> temp(row, inp.col);
    if(matrix.size() > 1000)
        dot_product(getData(), inp.getData(), temp.getData(), row, inp.row, inp.col);
    else
        dot(*this, inp, temp);
    return temp;
}

template <typename T>
void general_Scalar_operation(matrice<T>& a, T b, matrice<T>& dest, Operations ops)
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
void general_operation(matrice<T>& a, matrice<T>& b, matrice<T>& dest, Operations ops)
{
    for(int i = 0; i < a.numRows(); i++)
    {
        for(int j = 0; j < a.numCols(); j++)
        {
            int index = i * dest.numCols() + j;
            switch(ops)
            {
                case Add:
                    dest[index] = a[index] + b[index];
                    break;
                case Subtract:
                    dest[index] = a[index] - b[index];
                    break;
                case Multiply:
                    dest[index] = a[index] * b[index];
                    break;
                case Division:
                    dest[index] = a[index] / b[index];
            }
        }
    }
}


template <typename T>
void general_vector_operation(matrice<T>& a, matrice<T>& b, matrice<T>& dest, Operations ops, int axis)
{
    for(int i = 0; i < a.numRows(); i++)
    {
        for(int j = 0; j < a.numCols(); j++)
        {
            int index = i * dest.numCols() + j;
            int b_index = (axis) ? index - j: j;
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
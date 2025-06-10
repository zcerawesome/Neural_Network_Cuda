#ifndef MATRICE_GPU_H
#define MATRICE_GPU_H
#include <vector>
#include <iostream>
#include "matrice_helper.cuh"


template <typename T>
class matrice_gpu
{
private:
    int row, col;
public:
    std::vector<T> matrix;
    std::vector<int> shape();
    void resize(int row, int col);
    void toString();
    int numRows();
    T& get(int x, int y);
    int numCols();
    matrice_gpu();
    matrice_gpu(int row, int col);
    matrice_gpu(std::vector<T>& inp);
    matrice_gpu(const std::vector<T>& inp);
    matrice_gpu(std::vector<std::vector<T>>& inp);
    matrice_gpu(const std::vector<std::vector<T>>& inp);
    matrice_gpu<T> transpose();
    matrice_gpu<T> operator=(std::vector<std::vector<T>>& inp);
    matrice_gpu<T> operator=(const std::vector<std::vector<T>>& inp);
    matrice_gpu<T> operator=(std::vector<T>& inp);
    matrice_gpu<T> operator=(const std::vector<T>& inp);

    void operator=(matrice_gpu<T>& inp);
    void operator=(const matrice_gpu<T>& inp);
    void update(int row, int col);

    T& operator[](int row);
    T* getData();
    T sum();
    T max();

    matrice_gpu<T> getRows(int start, int end);
    matrice_gpu<T> getCols(int start, int end);

    matrice_gpu<T> operator-(matrice_gpu<T>& inp);
    matrice_gpu<T> operator-(const matrice_gpu<T>& inp);
    matrice_gpu<T> operator-(T inp);

    matrice_gpu<T> operator+(matrice_gpu<T>& inp);
    matrice_gpu<T> operator+(const matrice_gpu<T>& inp);
    matrice_gpu<T> operator+(T inp);

    matrice_gpu<T> operator*(matrice_gpu<T>& inp);
    matrice_gpu<T> operator*(const matrice_gpu<T>& inp);
    matrice_gpu<T> operator*(T inp);

    matrice_gpu<T> operator/(T inp);
    inline void dot(matrice_gpu<T>& a, matrice_gpu<T>& b, matrice_gpu<T>& dest);

    matrice_gpu<T> Dot(const matrice_gpu<T>& inp);
    matrice_gpu<T> Dot(matrice_gpu<T>& inp);
    
};

template <typename T>
void general_Scalar_operation(matrice_gpu<T>& a, T b, matrice_gpu<T>& dest, Operations ops);

template <typename T>
void general_operation(matrice_gpu<T>& a, matrice_gpu<T>& b, matrice_gpu<T>& dest, Operations ops);


#include "../src/matrice_gpu.cpp"
#endif
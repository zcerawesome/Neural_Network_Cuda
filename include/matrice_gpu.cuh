#ifndef MATRICE_GPU_CUH
#define MATRICE_GPU_CUH
#include <vector>
#include <iostream>
#include "matrice_helper.cuh"


template <typename T>
class matrice_gpu
{
private:
    int row, col;
public:
    T* matrix = nullptr;
    std::string getShape();
    Dim2 shape();
    void resize(int row, int col);
    void toString();
    int numRows();
    T get(int x, int y);
    std::vector<T> CPU_data();
    int numCols();
    int size();
    matrice_gpu();
    matrice_gpu(matrice_gpu<T>& inp);
    matrice_gpu(const matrice_gpu<T>& inp);
    matrice_gpu(int row, int col);
    matrice_gpu(std::vector<T>& inp);
    matrice_gpu(const std::vector<T>& inp);
    matrice_gpu(std::vector<std::vector<T>>& inp);
    matrice_gpu(const std::vector<std::vector<T>>& inp);
    ~matrice_gpu();
    matrice_gpu<T> transpose();
    matrice_gpu<T> operator=(std::vector<std::vector<T>>& inp);
    matrice_gpu<T> operator=(const std::vector<std::vector<T>>& inp);
    matrice_gpu<T> operator=(std::vector<T>& inp);
    matrice_gpu<T> operator=(const std::vector<T>& inp);
    matrice_gpu<T> operator=(T inp);
    

    void operator=(matrice_gpu<T>& inp);
    void operator=(const matrice_gpu<T>& inp);
    void update(int row, int col);

    T& operator[](int row);
    T* getData();
    T sum();
    T max();
    int largest_index(int col);

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

    matrice_gpu<T> Dot(const matrice_gpu<T>& inp);
    matrice_gpu<T> Dot(matrice_gpu<T>& inp);
    
    T* begin();
    T* end();

    const T* begin() const;
    const T* end() const;
};

template <typename T>
void general_Scalar_operation(matrice_gpu<T>& a, T b, matrice_gpu<T>& dest, Operations ops);

template <typename T>
void general_operation(matrice_gpu<T>& a, matrice_gpu<T>& b, matrice_gpu<T>& dest, Operations ops);


#include "../src/matrice_gpu.cu"
#endif
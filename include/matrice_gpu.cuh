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
    void update(int row, int col);
    std::string toString();
    int numRows();
    int numCols();
    T get(int x, int y);
    std::vector<T> CPU_data();
    int size();
    void deletePointer() const;   
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
    void operator=(std::vector<std::vector<T>>& inp);
    void operator=(const std::vector<std::vector<T>>& inp);
    void operator=(std::vector<T>& inp);
    void operator=(const std::vector<T>& inp);
    void operator=(T inp);
    

    void operator=(matrice_gpu<T>& inp);
    void operator=(const matrice_gpu<T>& inp);

    T& operator[](int row);
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
    
};

template <typename T>
int num_correct(matrice_gpu<T>& output, matrice_gpu<T>& answer);

#include "../src/matrice_gpu.cu"
#endif
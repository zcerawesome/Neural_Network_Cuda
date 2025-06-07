#ifndef MATRICE_H
#define MATRICE_H
#include <vector>
#include <iostream>
#include "matrice_helper.cuh"


template <typename T>
class matrice
{
private:
    int row, col;
public:
    std::vector<T> matrix;
    void resize(int row, int col);
    void toString();
    int numRows();
    T& get(int x, int y);
    int numCols();
    matrice();
    matrice(int row, int col);
    matrice(std::vector<T>& inp);
    matrice(const std::vector<T>& inp);
    matrice(std::vector<std::vector<T>>& inp);
    matrice(const std::vector<std::vector<T>>& inp);
    matrice<T> transpose();
    matrice<T> operator=(std::vector<std::vector<T>>& inp);
    matrice<T> operator=(const std::vector<std::vector<T>>& inp);
    matrice<T> operator=(std::vector<T>& inp);
    matrice<T> operator=(const std::vector<T>& inp);

    void operator=(matrice<T>& inp);
    void operator=(const matrice<T>& inp);
    void update(int row, int col);

    T& operator[](int row);
    T* getData();
    T sum();
    T max();

    matrice<T> getRows(int start, int end);
    matrice<T> getCols(int start, int end);

    matrice<T> operator-(matrice<T>& inp);
    matrice<T> operator-(const matrice<T>& inp);
    matrice<T> operator-(T inp);

    matrice<T> operator+(matrice<T>& inp);
    matrice<T> operator+(const matrice<T>& inp);
    matrice<T> operator+(T inp);

    matrice<T> operator*(matrice<T>& inp);
    matrice<T> operator*(const matrice<T>& inp);
    matrice<T> operator*(T inp);

    matrice<T> operator/(T inp);
    inline void dot(matrice<T>& a, matrice<T>& b, matrice<T>& dest);

    matrice<T> Dot(const matrice<T>& inp);
    matrice<T> Dot(matrice<T>& inp);

};

template <typename T>
void general_Scalar_operation(matrice<T>& a, T b, matrice<T>& dest, Operations ops);

template <typename T>
void general_operation(matrice<T>& a, matrice<T>& b, matrice<T>& dest, Operations ops);

#include "../src/matrice.cpp"
#endif
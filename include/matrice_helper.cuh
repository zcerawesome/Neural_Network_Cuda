#pragma once
#include <cuda_runtime.h>
#include <iostream>

enum Operations
{
    Add,
    Subtract,
    Multiply,
    Division
};

struct Dim2 
{
    int row, col;
    int& operator[](int val)
    {
        return *(&row + val);
    }
};

template <typename T>
void General_operation_helper(const T* a, const T* b, T* dest, Operations op, Dim2 a_dim, Dim2 b_dim);

template <typename T>
void General_scalar_helper(const T* a, T scalar, T* dest, int n, Operations op);

template <typename T>
void dot_product(const T* a, const T* b, T* dest, int firstRow, int secondRow, int lastCol);

template <typename T>
void transpose_GPU(const T* a, T* dest, Dim2 a_dim);

template <typename T>
void largest_index(const T* a, T* dest, Dim2 a_dim);

template <typename T>
void general_clamp(const T* a, T* dest, int size, T lower, T higher);
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

template <typename T>
void General_operation_helper(const T* a, const T* b, T* dest, int n, Operations op);

template <typename T>
void General_scalar_helper(const T* a, T scalar, T* dest, int n, Operations op);

template <typename T>
void dot_product(const T* a, const T* b, T* dest, int firstRow, int secondRow, int lastCol);

template <typename T>
void sum_cuda(const T* a, T* dest, int size);
#ifndef NETWORK_FUNCTIONS_CUH
#define NETWORK_FUNCTIONS_CUH
#include <vector>
#include <iostream>
#include "matrice_helper.cuh"
#include "matrice_gpu.cuh"
#include <curand.h>
#include <curand_kernel.h>


void randomize_matrix(matrice_gpu<float>& inp);
matrice_gpu<float> ReLU(matrice_gpu<float>& inp);
matrice_gpu<float> ReLU_derive(matrice_gpu<float>& inp);
matrice_gpu<float> softmax(matrice_gpu<float>& inp);
matrice_gpu<float> one_hot_encode(matrice_gpu<float>& y);

#endif
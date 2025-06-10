#ifndef LAYER_H
#define LAYER_H
#include "matrice_gpu.h"

struct Layer
{
    matrice_gpu<float> weight;
    matrice_gpu<float> bias;
    matrice_gpu<float> (*activation_function)(matrice_gpu<float>& inp);
    matrice_gpu<float> (*activation_function_derive)(matrice_gpu<float>& inp);
    int outputs;
};

#endif
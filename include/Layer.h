#ifndef LAYER_H
#define LAYER_H
#include "matrice.h"

struct Layer
{
    matrice<float> weight;
    matrice<float> bias;
    matrice<float> (*activation_function)(matrice<float>& inp);
    matrice<float> (*activation_function_derive)(matrice<float>& inp);
    int outputs;
};

#endif
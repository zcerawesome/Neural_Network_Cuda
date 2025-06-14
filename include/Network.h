#ifndef NETWORK_H
#define NETWORK_H
#define vec(X) std::vector<X>
#define vec2D(X) std::vector<std::vector<X>>
#include <vector>
#include "Layer.h"
#include <math.h>
#include "Network_functions.cuh"

class Network
{
private:
    void (*Weight_randomization)(matrice_gpu<float>&) = 0;
public:
    std::vector<Layer> layers;   
    void addLayer(int output, matrice_gpu<float> (*activation_function)(matrice_gpu<float>& inp), matrice_gpu<float> (*activation_function_derive)(matrice_gpu<float>& inp));
    void setRandomization(void (*Weight_randomization)(matrice_gpu<float>&));
    void applyRandomzation(int layer);
    vec(matrice_gpu<float>) forward(matrice_gpu<float>& X);
    vec(matrice_gpu<float>) backward_prop(vec(matrice_gpu<float>)& forward, matrice_gpu<float>& X, matrice_gpu<float>& Y);
    void update_params(vec(matrice_gpu<float>)& back_prop, float alpha);
};

#include "../src/Network.cpp"

#endif
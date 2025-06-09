#ifndef NETWORK_H
#define NETWORK_H
#define vec(X) std::vector<X>
#define vec2D(X) std::vector<std::vector<X>>
#include <vector>
#include "Layer.h"
#include <math.h>


class Network
{
private:
    void (*Weight_randomization)(matrice<float>&) = 0;
public:
    std::vector<Layer> layers;   
    void addLayer(int output, matrice<float> (*activation_function)(matrice<float>& inp), matrice<float> (*activation_function_derive)(matrice<float>& inp));
    void setRandomization(void (*Weight_randomization)(matrice<float>&));
    void applyRandomzation(int layer);
    matrice<float> one_hot_encode(matrice<float>& y);
    vec(matrice<float>) forward(matrice<float>& X);
    vec(matrice<float>) backward_prop(vec(matrice<float>)& forward, matrice<float>& X, matrice<float>& Y);
    void update_params(vec(matrice<float>)& back_prop, float alpha);
};

#include "../src/Network.cpp"

#endif
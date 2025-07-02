#ifndef NETWORK_H
#define NETWORK_H
#define vec(X) std::vector<X>
#define vec2D(X) std::vector<std::vector<X>>
#include <vector>
#include "Layer.h"
#include <math.h>
#include "Network_functions.cuh"

template <typename T>
class Network
{
private:
    void (*Weight_randomization)(matrice_gpu<T>&) = 0;
public:
    std::vector<Layer> layers;   
    void addLayer(int output, matrice_gpu<T> (*activation_function)(matrice_gpu<T>& inp), matrice_gpu<T> (*activation_function_derive)(matrice_gpu<T>& inp));
    void load_parameters(std::string filename, bool header=true);
    void setRandomization(void (*Weight_randomization)(matrice_gpu<T>&));
    void applyRandomzation(int layer);
    vec(matrice_gpu<T>) forward(matrice_gpu<T>& X);
    vec(matrice_gpu<T>) backward_prop(vec(matrice_gpu<T>)& forward, matrice_gpu<T>& X, matrice_gpu<T>& Y);
    void update_params(vec(matrice_gpu<T>)& back_prop, float alpha);

    void train(int epochs, float alpha, matrice_gpu<T>& X_train, matrice_gpu<T>& Y_train, matrice_gpu<float>& Y, bool print = true);
    void test(matrice_gpu<T>& X_test, matrice_gpu<T>& Y_test);
    void save_data(std::string file_name);
};

#include "../src/Network.cpp"

#endif
#include "Network.h"

void Network::addLayer(int output, matrice_gpu<float> (*activation_function)(matrice_gpu<float>& inp), matrice_gpu<float> (*activation_function_derive)(matrice_gpu<float>& inp))
{
    if(!layers.size())
    {
        layers.push_back({{}, {}, 0, 0, output});
        return;
    }
    Layer& last_layer = layers[layers.size()-1];
    layers.push_back({{output, last_layer.outputs}, {output, 1}, activation_function, activation_function_derive, output});
}


void Network::setRandomization(void (*Weight_randomization)(matrice_gpu<float>&))
{
    this->Weight_randomization = Weight_randomization;
}


void Network::applyRandomzation(int layer)
{
    Layer& picked_layer = layers[layer];
    Weight_randomization(picked_layer.weight);
    Weight_randomization(picked_layer.bias);
}

bool has_nan(matrice_gpu<float>& inp)
{
    std::vector<float> cpu = inp.CPU_data();
    for(auto& value: cpu)
        if(isnan(value))
            return true;
    return false;
}

bool has_nan(vec(matrice_gpu<float>)& inp)
{
    for(auto& matrix: inp)
    {
        std::vector<float> cpu = matrix.CPU_data();
        for(auto& value: cpu)
            if(isnan(value))
                return true;
    }
    return false;
}

vec(matrice_gpu<float>) Network::forward(matrice_gpu<float>& X)
{
    vec(matrice_gpu<float>) output((layers.size() - 1) * 2);
    int j = -1;
    for(int i = 0; i < layers.size()-1; i++)
    {
        Layer& layer = layers[i + 1];
        matrice_gpu<float>& Z = output[i * 2];
        matrice_gpu<float>& A = output[i * 2 + 1];
        if(i == 0)
            Z = layer.weight.Dot(X) + layer.bias;
        else
            Z = layer.weight.Dot(output[i * 2 - 1]) + layer.bias;
        A = layer.activation_function(Z);
    }
    return output;
}

vec(matrice_gpu<float>) Network::backward_prop(vec(matrice_gpu<float>)& forward, matrice_gpu<float>& X, matrice_gpu<float>& Y)
{
    int col = Y.numCols();
    auto one_hot_encode_y = one_hot_encode(Y);

    vec(matrice_gpu<float>) results((layers.size() - 1) * 2);
    vec(matrice_gpu<float>) not_results(layers.size()-1);
    int iteration = 0;
    for(int i = layers.size() - 2; i >= 0; i--)
    {
        if(i == layers.size()-2)
            not_results[i]  = forward.back() - one_hot_encode_y;   
        else
            not_results[i] = layers[i+2].weight.transpose().Dot(not_results[i+1]) * layers[i+1].activation_function_derive(forward[i*2]);
        if(i == 0)
            results[i * 2] = not_results[i].Dot(X.transpose()) / col;
        else
            results[i * 2] = not_results[i].Dot(forward[i * 2 - 1].transpose()) / col;
        results[i * 2 + 1] = not_results[i].sum() / col;
    }
    return results;
}


void Network::update_params(vec(matrice_gpu<float>)& back_prop, float alpha)
{
    for(int i = 0; i < back_prop.size()/2; i++)
    {
        Layer& layer = layers[i + 1];
        layer.weight = layer.weight - back_prop[i * 2] * alpha;
        layer.bias = layer.bias - (back_prop[i * 2 + 1] * alpha).get(0, 0);
    }
}
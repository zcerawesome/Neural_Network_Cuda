#include "Network.h"

void Network::addLayer(int output, matrice<float> (*activation_function)(matrice<float>& inp), matrice<float> (*activation_function_derive)(matrice<float>& inp))
{
    if(!layers.size())
    {
        layers.push_back({{}, {}, 0, 0, output});
        return;
    }
    Layer& last_layer = layers[layers.size()-1];
    layers.push_back({{output, last_layer.outputs}, {output, 1}, activation_function, activation_function_derive, output});
}


void Network::setRandomization(void (*Weight_randomization)(matrice<float>&))
{
    this->Weight_randomization = Weight_randomization;
}


void Network::applyRandomzation(int layer)
{
    Layer& picked_layer = layers[layer];
    Weight_randomization(picked_layer.weight);
    Weight_randomization(picked_layer.bias);
}

matrice<float> Network::one_hot_encode(matrice<float>& y)
{
    matrice<float> one_hot(y.max() + 1, y.numCols());
    for(int i = 0; i < y.numCols(); i++)
        one_hot.get(y.get(0,i), i) = 1.0f;
    return one_hot;
}

bool has_nan(matrice<float>& inp)
{
    for(auto& value: inp.matrix)
        if(isnan(value))
            return true;
    return false;
}

bool has_nan(vec(matrice<float>)& inp)
{
    for(auto& matrix: inp)
        for(auto& value: matrix.matrix)
            if(isnan(value))
                return true;
    return false;
}

bool val_greater_than(matrice<float>& inp, float value)
{
    
    return false;
}

vec(matrice<float>) Network::forward(matrice<float>& X)
{
    vec(matrice<float>) output((layers.size() - 1) * 2);
    int j = -1;
    for(int i = 0; i < layers.size()-1; i++)
    {
        Layer& layer = layers[i + 1];
        matrice<float>& Z = output[i * 2];
        matrice<float>& A = output[i * 2 + 1];
        if(i == 0)
            Z = layer.weight.Dot(X) + layer.bias;
        else
            Z = layer.weight.Dot(output[i * 2 - 1]) + layer.bias;
        A = layer.activation_function(Z);
        // std::cout << "Iteration: " << ++j << std::endl;
        // std::cout << "Z: " << has_nan(Z) << std::endl;
        // std::cout << "A: " << has_nan(A) << std::endl;
        // std::cout << "Dot CPU" << std::endl;
    }
    
    for(auto& matrix: output)
        std::cout <<  matrix.max() << std::endl;
    exit(0);
    return output;
}

vec(matrice<float>) Network::backward_prop(vec(matrice<float>)& forward, matrice<float>& X, matrice<float>& Y)
{
    int col = Y.numCols();
    auto one_hot_encode_y = one_hot_encode(Y);

    vec(matrice<float>) results((layers.size() - 1) * 2);
    vec(matrice<float>) not_results(layers.size()-1);
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
        results[i * 2 + 1].matrix = {{not_results[i].sum() / col}};
        results[i * 2 + 1].update(1, 1);
    }
    return results;
}


void Network::update_params(vec(matrice<float>)& back_prop, float alpha)
{
    for(int i = 0; i < back_prop.size()/2; i++)
    {
        Layer& layer = layers[i + 1];
        layer.weight = layer.weight - back_prop[i * 2] * alpha;
        layer.bias = layer.bias - (back_prop[i * 2 + 1] * alpha)[0];
    }
}
#include "Network.h"

template <typename T>
void Network<T>::addLayer(int output, matrice_gpu<T> (*activation_function)(matrice_gpu<T>& inp), matrice_gpu<T> (*activation_function_derive)(matrice_gpu<T>& inp))
{
    if(!layers.size())
    {
        layers.push_back({{}, {}, 0, 0, output});
        return;
    }
    Layer& last_layer = layers[layers.size()-1];
    layers.push_back({{output, last_layer.outputs}, {output, 1}, activation_function, activation_function_derive, output});
}

template <typename T>
void Network<T>::load_parameters(std::string filename, bool header)
{
    int counter = 0;
    for(int i = 1; i < layers.size(); i++)
    {
        Layer& layer = layers[i];
        layer.weight.load_data(filename, header, layer.weight.numRows(), counter);
        header = false;
        counter += layer.weight.numRows();
        layer.bias.load_data(filename, header, layer.bias.numRows(), counter);
        counter += layer.bias.numRows();
    }
}

template <typename T>
void Network<T>::setRandomization(void (*Weight_randomization)(matrice_gpu<T>&))
{
    this->Weight_randomization = Weight_randomization;
}

template <typename T>
void Network<T>::applyRandomzation(int layer)
{
    Layer& picked_layer = layers[layer];
    Weight_randomization(picked_layer.weight);
    Weight_randomization(picked_layer.bias);
}

template <typename T>
bool has_nan(matrice_gpu<T>& inp)
{
    std::vector<T> cpu = inp.CPU_data();
    for(auto& value: cpu)
        if(isnan(value))
            return true;
    return false;
}

template <typename T>
bool has_nan(vec(matrice_gpu<T>)& inp)
{
    for(auto& matrix: inp)
    {
        std::vector<T> cpu = matrix.CPU_data();
        for(auto& value: cpu)
            if(isnan(value))
                return true;
    }
    return false;
}

template <typename T>
vec(matrice_gpu<T>) Network<T>::forward(matrice_gpu<T>& X)
{
    vec(matrice_gpu<T>) output((layers.size() - 1) * 2);
    int j = -1;
    for(int i = 0; i < layers.size()-1; i++)
    {
        Layer& layer = layers[i + 1];
        matrice_gpu<T>& Z = output[i * 2];
        matrice_gpu<T>& A = output[i * 2 + 1];
        if(i == 0)
            Z = layer.weight.Dot(X) + layer.bias;
        else
            Z = layer.weight.Dot(output[i * 2 - 1]) + layer.bias;
        A = layer.activation_function(Z);
    }
    return output;
}

template <typename T>
vec(matrice_gpu<T>) Network<T>::backward_prop(vec(matrice_gpu<T>)& forward, matrice_gpu<T>& X, matrice_gpu<T>& one_hot_encode_Y)
{
    int col = one_hot_encode_Y.numCols();

    vec(matrice_gpu<T>) results((layers.size() - 1) * 2);
    vec(matrice_gpu<T>) not_results(layers.size()-1);
    int iteration = 0;
    for(int i = layers.size() - 2; i >= 0; i--)
    {
        if(i == layers.size()-2)
            not_results[i]  = forward.back() - one_hot_encode_Y;   
        else
            not_results[i] = layers[i+2].weight.transpose().Dot(not_results[i+1]) * layers[i+1].activation_function_derive(forward[i*2]);
        if(i == 0)
            results[i * 2] = not_results[i].Dot(X.transpose()) / col;
        else
            results[i * 2] = not_results[i].Dot(forward[i * 2 - 1].transpose()) / col;
        results[i * 2 + 1] = not_results[i].sum_axis(1) / col;
    }
    return results;
}

template <typename T>
void Network<T>::update_params(vec(matrice_gpu<T>)& back_prop, float alpha)
{
    for(int i = 0; i < back_prop.size()/2; i++)
    {
        Layer& layer = layers[i + 1];
        layer.weight = layer.weight - back_prop[i * 2] * alpha;
        layer.bias = layer.bias - (back_prop[i * 2 + 1] * alpha);
    }
}

template <typename T>
void Network<T>::train(int epochs, float alpha, matrice_gpu<T>& X_train, matrice_gpu<T>& Y_train, matrice_gpu<float>& Y, bool print)
{
    if(print)
        std::cout << "Starting Clock" << std::endl;
    clock_t start = clock();
    int total = Y_train.numCols();
    for(int i = 0; i < epochs; i++)
    {
        vec(matrice_gpu<T>) results = forward(X_train);
        vec(matrice_gpu<T>) dds = backward_prop(results, X_train, Y_train);
        update_params(dds, alpha);
        if(print && (i + 1 ) % 10 == 0)
        {
            // int total_correct = num_correct(results[3], Y);
            std::cout << "Iteration " << (i+1) << std::endl;
            // std::cout << "Accuracy " << ((float)(total_correct) / total) << std::endl;
        }
    }
    clock_t end = clock();
    if(print)
        std::cout << "Simulation Time: " << (end - start) / 1000000.0 << std::endl;
}

template <typename T>
void Network<T>::test(matrice_gpu<T>& X_test, matrice_gpu<T>& Y_test)
{
    vec(matrice_gpu<T>) results = forward(X_test);
    int correct = num_correct(results[3], Y_test);
    std::cout << "Accuracy: " << (float)correct / Y_test.numCols() << std::endl;
}

template <typename T>
void Network<T>::save_data(std::string file_name)
{
    std::ofstream net(file_name);
    if(!net.is_open())
    {
        std::cerr << "Error, cannot open file to save data" << std::endl;
        return;
    }
    for(int i = 1; i < layers.size(); i++)
    {
        Layer& layer = layers[i];
        net << layer.weight.toString();
        net << layer.bias.toString();
    }
    net.close();
}
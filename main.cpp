#include <iostream>
#include "matrice_gpu.cuh"
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include "Network.h"

#define vec(X) std::vector<X>
#define vec2D(X) std::vector<std::vector<X>>

std::vector<std::vector<float>> loadCSV(std::string fileName, bool header=true, int maxRows=1000, int row_number=0)
{
    std::vector<std::vector<float>> data;
    std::ifstream file(fileName);
    if(!file.is_open())
        std::cerr << "Error file path is wrong" << std::endl;
    std::string line;
    if(header)
        getline(file, line);
    int i = 0;
    while(i < row_number && getline(file, line))
        i++;
    i = 0;
    while(getline(file, line) && (i < maxRows || maxRows == -1))
    {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;

        while(std::getline(ss, value, ','))
        {
            row.push_back(std::stod(value));
        }
        data.push_back(row);
        i++;
    }
    return data;
}

int largest_index(std::vector<float>& input, int row, int numColumn, int numRow)
{
    int index = 0;
    for(int i = 0; i < numRow; i++)
        index = input[i * numColumn + row] > input[index * numColumn + row]? i: index;
    return index;
}

int main()
{
    Network network;
    network.addLayer(784, 0, 0);
    network.addLayer(10, ReLU, ReLU_derive);
    network.addLayer(10, softmax, 0);

    network.setRandomization(randomize_matrix);
    network.applyRandomzation(1);
    network.applyRandomzation(2);

    std::string fileName = "../data/train.csv";
    std::vector<std::vector<float>> df = loadCSV(fileName, true, -1);
    matrice_gpu<float> data(df);

    data = data.transpose();
    int rows = data.numRows();
    int cols = data.numCols();

    matrice_gpu<float> Train = data.getCols(1000, cols);
    matrice_gpu<float> X_train = Train.getRows(1,rows);
    X_train = X_train / 255.0;
    matrice_gpu<float> Y_train = Train.getRows(0,1);

    matrice_gpu<float> test = data.getCols(0, 1000);
    matrice_gpu<float> X_test = test.getRows(1, rows);
    X_test = X_test / 255.0;
    matrice_gpu<float> Y_test = test.getRows(0,1);

    std::vector<float> Y_train_cpu = Y_train.CPU_data();
    std::cout << "Starting Clock" << std::endl;
    clock_t timer_start = clock();
    int total = X_train.numCols();
    for(int i = 0; i < 500; i++)
    {
        vec(matrice_gpu<float>) results = network.forward(X_train);
        vec(matrice_gpu<float>) dds = network.backward_prop(results, X_train, Y_train);
        network.update_params(dds, .1);
        
        if((i+1) % 10 == 0)
        {
            int total_correct = num_correct(results[3], Y_train);
            std::cout << "Iteration " << (i+1) << std::endl;
            std::cout << "Accuracy " << ((float)(total_correct) / total) << std::endl;
        }
    }
    clock_t timer_end = clock();
    std::cout << "Simulation Time: " << (timer_end - timer_start) << std::endl;
    vec(matrice_gpu<float>) results = network.forward(X_test);
    int correct = num_correct(results[3], Y_test);
    std::cout << "Accuracy: " << (float)correct / Y_test.numCols() << std::endl;
}
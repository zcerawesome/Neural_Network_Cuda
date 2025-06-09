#include <iostream>
#include "matrice_gpu.h"
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

float randomFloat()
{
    return (float)rand() / RAND_MAX;
}

void randomize_matrix(matrice<float>& inp)
{
    for(auto& rows: inp.matrix)
        rows = randomFloat() - 0.5;
}

matrice<float> ReLU(matrice<float>& inp)
{
    matrice<float> temp(inp.numRows(), inp.numCols());
    for(int i = 0; i < temp.matrix.size(); i++)
        temp[i] = inp[i] > 0? inp[i]: 0;
    return temp;
}

matrice<float> ReLU_derive(matrice<float>& inp)
{
    matrice<float> temp(inp.numRows(), inp.numCols());
    for(int i = 0; i < temp.matrix.size(); i++)
        temp[i] = inp[i] > 0;
    return temp;
}

matrice<float> softmax(matrice<float>& inp)
{
    matrice<float> temp(inp.numRows(), inp.numCols());
    for(int j = 0; j < temp.numCols(); j++)
    {
        float sum = 0;
        for(int i = 0; i < temp.numRows(); i++)
        {
            temp.get(i,j) = exp(inp.get(i, j));
            sum += temp.get(i, j);
            
        }
        for(int i = 0; i < temp.numRows(); i++)
        {
            temp.get(i,j) /= sum;
            // if(isnan(temp.get(i,j) / sum))
            // {
            //     std::cerr << "EXP is nan" << std::endl;
            //     std::cout << inp.get(i, j) << std::endl;
            //     std::cout << sum << std::endl;
            //     exit(0);
            // }
        }
    }
    // if(has_nan(temp))
    // {
    //     std::cerr << "softmax has nan" << std::endl;
    //     exit(0);
    // }
    return temp;
}

int largest_index(matrice<float>& input, int row = 0)
{
    int index = 0;
    for(int i = 0; i < input.numRows(); i++)
        index = input.get(i, row) > input.get(index, row)? i: index;
    return index;
}

int main()
{
    /*
    int vec[1000];
    for(int i = 0 ; i < sizeof(vec) / sizeof(int); i++)
        vec[i] = i;
    clock_t time1 = clock();
    int sum = 0;
    for(int i = 0 ; i < sizeof(vec) / sizeof(int); i++)
        sum += vec[i];
    clock_t time2 = clock();
    std::cout << (time2 - time1) << std::endl;
    std::cout << "CPU Sum: " << sum << std::endl;
    time1 = clock();
    sum = 0;
    sum_cuda(vec, &sum, sizeof(vec) / sizeof(int));
    time2 = clock();
    std::cout << (time2 - time1) << std::endl;
    std::cout << "GPU Sum: " << sum << std::endl;
    return 0;*/
    // srand(45);
    Network network;
    network.addLayer(784, 0, 0);
    network.addLayer(10, ReLU, ReLU_derive);
    network.addLayer(10, softmax, 0);

    network.setRandomization(randomize_matrix);
    network.applyRandomzation(1);
    network.applyRandomzation(2);
    std::string fileName = "../data/train.csv";
    std::vector<std::vector<float>> df = loadCSV(fileName, true, 1000);
    matrice<float> data(df);
    data = data.transpose();
    int rows = data.numRows();
    int cols = data.numCols();

    matrice<float> Train = data.getCols(200, cols);
    matrice<float> X_train = Train.getRows(1,rows);
    X_train = X_train / 255.0;
    matrice<float> Y_train = Train.getRows(0,1);

    matrice<float> test = data.getCols(0, 200);
    matrice<float> X_test = test.getRows(1, rows);
    X_test = X_test / 255.0;
    matrice<float> Y_test = test.getRows(0,1);

    int total_correct = 0;
    int total = 0;
    std::cout << "Starting Clock" << std::endl;
    clock_t timer_start = clock();
    for(int i = 0; i < 500; i++)
    {
        int j = 0;
        std::cout << "Iteration: " << i << std::endl;
        vec(matrice<float>) results = network.forward(X_train);
        if(has_nan(results))
        {
            std::cout << "Forward has nan" << std::endl; exit(0);
        }
        vec(matrice<float>) dds = network.backward_prop(results, X_train, Y_train);
        if(has_nan(dds))
        {
            std::cout << "backward has nan" << std::endl; exit(0);
        }
        network.update_params(dds, .1);
        for(int j = 0; j < results[3].numCols(); j++)
        {
            if(largest_index(results[3], j) == Y_train.get(0, j))
                total_correct++;
        }
        total += results[3].numCols();
        if((i+1) % 10 == 0)
        {
            std::cout << "Iteration " << (i+1) << std::endl;
            std::cout << "Accuracy " << ((float)total_correct / total) << std::endl;
        }
        // if(i)
        //     exit(0);
    }
    clock_t timer_end = clock();
    std::cout << "Simulation Time: " << (timer_end - timer_start) << std::endl;
    int correct = 0;
    vec(matrice<float>) results = network.forward(X_test);
    for(int i = 0; i < Y_test.numCols(); i++)
    {
        if(largest_index(results[3], i) == Y_test.get(0, i))
            correct++;
        // std::cout << Y_test[i] << " " << largest_index(results[3], i) << std::endl;
    }
    std::cout << "Accuracy: " << (float)correct / Y_test.numCols() << std::endl;
}
#include <iostream>
#include "matrice.h"
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
        }
    }
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
    srand(time(0)+45);
    Network network;
    network.addLayer(784, 0, 0);
    network.addLayer(10, ReLU, ReLU_derive);
    network.addLayer(10, softmax, 0);

    network.setRandomization(randomize_matrix);
    network.applyRandomzation(1);
    network.applyRandomzation(2);

    

}
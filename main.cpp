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

void generate_image()
{
    std::string fileName = "../data/train.csv";
    matrice_gpu<float> data;
    data.load_data(fileName, true, -1);
    data = data.transpose();
    int rows = data.numRows();
    int cols = data.numCols();

    matrice_gpu<float> Train = data.getCols(1000, cols);
    matrice_gpu<float> Y_train = Train.getRows(1,rows);
    Y_train = Y_train / 255.0;
    matrice_gpu<float> X_train = Train.getRows(0,1);

    matrice_gpu<float> test = data.getCols(0, 1000);
    matrice_gpu<float> Y_test = test.getRows(1, rows);
    Y_test = Y_test / 255.0;
    matrice_gpu<float> X_test = test.getRows(0,1);
    Network<float> network;
    network.addLayer(10, 0, 0);
    // network.addLayer(10, ReLU, ReLU_derive);
    network.addLayer(64, ReLU, ReLU_derive);
    network.addLayer(128, ReLU, ReLU_derive);
    network.addLayer(784, sigmoid, 0);
    // network.setRandomization(randomize_matrix);
    // network.applyRandomzation(1);
    // network.applyRandomzation(2);
    // network.applyRandomzation(3);
    for(int i = 1; i < 3; i++)
    {
        Layer& layer = network.layers[i];
        random_sample(layer.weight, 0, sqrt(2 / layer.weight.numCols()));
    }
    Layer& last_layer = network.layers[3];
    random_sample(last_layer.weight, 0, 2 / (last_layer.weight.numCols() + last_layer.weight.numRows()));
    X_train = one_hot_encode(X_train, 9);
    char train = 'y';
    int total_iterations = 0;
    while(train == 'y')
    {
        int i;
        std::cout << "Enter iterations to train: ";
        std::cin >> i;
        network.train(i, 0.1, X_train, Y_train, Y_train);
        std::vector<float> examples = {0,1,2,3,4,5,6,7,8,9};
        matrice_gpu<float> guess(examples);
        guess = one_hot_encode(guess, 9);
        vec(matrice_gpu<float>) forward = network.forward(guess);
        matrice_gpu<float> answer = forward.back().transpose();
        std::ofstream output("../output/Guess2.csv");
        // forward.back().update(1, 784);
        output << answer.toString();
        output.close();
        total_iterations += i;
        std::cout << "Total Iterations: " << total_iterations << std::endl;
    }
    // network.save_data("../output/MNIST_Digit_4_layer.txt");
    // network.load_parameters("../output/MNIST_Digit_4_layer.txt",false);
    
    // network.test(X_test, Y_test);
}

int main()
{
    // matrice_gpu<float> guess;
    // guess = 0;
    // guess = sigmoid(guess);
    // std::cout << guess.toString() << std::endl;
    // exit(0);
    generate_image();
    // matrice_gpu<float> temp(10, 10);
    // random_sample(temp, 0, 1);
    // std::cout << temp.toString() << std::endl;
    return 0;
    std::string fileName = "../data/train.csv";
    matrice_gpu<float> data;
    data.load_data(fileName, true, -1);
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
    Network<float> network;
    network.addLayer(784, 0, 0);
    network.addLayer(10, ReLU, ReLU_derive);
    network.addLayer(10, softmax, 0);

    network.setRandomization(randomize_matrix);
    char stored_data = 'n';
    // std::cout << "Want to use stored data? (Y/N) ";
    // std::cin >> stored_data;
    if(stored_data == 'Y' || stored_data == 'y')
        network.load_parameters("../output/MNIST_Digit.txt", false);
    else
    {
        network.applyRandomzation(1);
        network.applyRandomzation(2);
        matrice_gpu<float> Y_encode;
        Y_encode = one_hot_encode(Y_train, 9);
        network.train(500, 0.1, X_train, Y_encode, Y_train);
    }
    auto Y_test_encode = one_hot_encode(Y_test, 9);
    std::cout << Y_test_encode.getShape() << std::endl;
    network.test(X_test, Y_test_encode);
    
    // matrice_gpu<float> input;
    // input.load_data("../test.csv", false, -1);
    // forward = network.forward(input);
    // std::cout << "Guessed value: " << forward[3].largest_index(0) << " " << forward[3].get(2,0)<< std::endl;
    // exit(0);
    char save_data;
    std::cout << "Do you want to save data ? (Y/N) ";
    std::cin >> save_data;
    if(save_data == 'Y' || save_data == 'y')
    {
        network.save_data("../output/MNIST_Digit.txt");
    }
}
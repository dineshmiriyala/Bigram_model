"""This is the driver function for model"""

from model import bigram, neural_network
import os
import pickle

print("---------------Bigram model --------------")
print("-------Enter only numbers or float--------")

if not os.path.exists('Bigram_lookup.pkl'):
    open('Bigram_lookup.pkl', 'w')
    BigramModel = False
else:
    try:
        with open('Bigram_lookup.pkl' , 'rb') as file:
            BigramModel = pickle.load(file)
    except Exception:
        BigramModel = False



while True:
    lines = open('reddit_convos.txt', 'r').read().splitlines()
    model = int(input("\n\n\nSelect a option: \n 1: Bigram \n 2: Neural Network \n 3: END \n"))
    if model == 1:
        if not BigramModel:
            BigramModel = bigram(lines)
            print(f"Model loss: {BigramModel.loss()}")
            with open('Bigram_lookup.pkl' , 'wb') as file:
                pickle.dump(BigramModel , file)
                print(">>>>>>>>>>Values Updated<<<<<<<<<< \n")
        print(">>>>>>>>>>Model loaded<<<<<<<<<< \n")
        n_lines = int(input('Enter how many lines you want to generate: \n'))
        BigramModel.generate(n_lines)

    elif model == 2:
        user = int(input("Enter what you want to do next: \n 1. Train \n 2. Generate text \n 3. Current loss \n"))
        NeuralNetworkModel = neural_network(lines)
        while True:
            if user == 1:
                if NeuralNetworkModel.loss_value:
                    print(f"Current Loss: {NeuralNetworkModel.loss_value:.4f} \n")
                    cont = input(f'Do you still want to continue training? (Y/N): \n')
                    if cont.lower() == 'n':
                        break
                print("Training Parameters: \n")
                iter = int(input("Enter the number of training iterations: \n"))
                NeuralNetworkModel.train(iter)
                user = int(input("Enter what you want to do next: \n 1. Train \n 2. Generate text \n"))
            elif user == 2:
                if NeuralNetworkModel.loss_value is None:
                    print("Model Not Found, need to train first. \n")
                    cont2 = input("Do you want to train now? (Y/N): \n")
                    if cont2.lower() == 'n':
                        break
                    else:
                        iter = int(input("Enter the number of training iterations: \n"))
                        NeuralNetworkModel.train(iter)
                    break
                n_lines = int(input('Enter how many lines you want to generate: \n'))
                NeuralNetworkModel.generate(n_lines)
                break
            else:
                if NeuralNetworkModel.loss_value:
                    print(f"Current Loss: {NeuralNetworkModel.loss_value:.4f} \n")
                    user = int(
                        input("Enter what you want to do next: \n 1. Train \n 2. Generate text \n"))
                else:
                    print("Model is not trained before. \n")
                    cont3 = input("Do you want to train now? (Y/N): \n")
                    if cont3.lower() == 'n':
                        break
                    else:
                        iter = int(input("Enter the number of training iterations: \n"))
                        NeuralNetworkModel.train(iter)
                    break
    else:
        break
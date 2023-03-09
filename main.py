"""This is the driver function for model"""

from model import bigram,neural_network
import os

print("---------------Bigram model --------------")
print("-------Enter only numbers or float--------")
lines = open('reddit_convos.txt' , 'r').read().splitlines()
model = int(input("Select model: \n 1: Bigram \n 2: Neural Network \n"))
os.system('cls')
if model == 1:
    n_lines = int(input('Enter how many lines you want to generate: \n'))
    os.system('cls')
    bigram = bigram(lines)
    bigram.generate(n_lines)
    print(bigram.loss())
else:
    user = 1
    while True:

        if user ==1:
            print("Training Parameters: \n")
            lr = float(input("Enter the learning rate: \n"))
            reg = float(input("Enter Regularization constant: \n"))
            iter = int(input("Enter the number of training iterations: \n"))
            os.system('cls')
            network = neural_network(lines, lr, reg, iter)
            network.train()
            user = int(input("Enter what you want to do next: \n 1. Train \n 2. Generate text \n"))
        else:
            n_lines = int(input('Enter how many lines you want to generate: \n'))
            network.generate(n_lines)
            break
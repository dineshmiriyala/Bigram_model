# imports
import os.path

import torch
import warnings
import torch.nn.functional as F
import pickle

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


class bigram():
    """Bi gram model is simply a lookup table of logits for the next character given the previous character."""

    def __init__(self, lines):
        """This has all the variables for bi gram model. This might not be the best approach to do this.
        But it does the job"""
        self.lines = lines
        char = sorted(list((set(''.join(self.lines)))))
        self.stoi = {s: i + 1 for i, s in enumerate(char)}
        self.stoi['.'] = 0
        self.itos = {i: s for s, i in self.stoi.items()}
        self.lookup = torch.zeros((len(self.stoi), len(self.stoi)), dtype=torch.int64)
        self.lookup_table()

    def lookup_table(self):
        """This function creates a character level lookup matrix.
        This does not return anything but does operations on the existing 'self.lookup' which is a
        torch. tensor with datatype as int 64"""
        length = len(self.lines)
        bar = length // 10
        progress = 0
        print("Progress: ", end='')
        for line in self.lines:
            if self.lines.index(line) % bar == 0:
                print(f'{progress}#', end='')
                progress += 10
            line = ['.'] + list(line) + ['.']
            for ch1, ch2 in zip(line, line[1:]):
                idx1 = self.stoi[ch1]
                idx2 = self.stoi[ch2]
                self.lookup[idx1, idx2] += 1
        print('\n')

    def plot(self, x_dimension, y_dimension):
        """Upon calling this function it plots the plot for lookup table. This takes
        x_dimension and y_dimension as arguments"""
        plt.figure(figsize=(x_dimension, y_dimension))
        plt.imshow(self.lookup, cmap='Oranges')
        for i in range(len(self.stoi)):
            for j in range(len(self.stoi)):
                chstr = self.itos[i] + '-' + self.itos[j]
                plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
                plt.text(j, i, self.lookup[i, j].item(), ha='center', va='top', color='gray')
        plt.axis('off')
        plt.title('Look up Table')
        plt.show()

    def generate(self, length):
        """This is the generator function. It takes the number of iterations as input and generates
        list of strings. It returns a list of strings."""

        generator = torch.Generator().manual_seed(9968)
        for iterations in range(length):
            output = []
            index = 0
            while True:
                probability = self.lookup[index].float()
                probability = probability / probability.sum()
                index = torch.multinomial(probability, num_samples=1, replacement=True, generator=generator).item()
                output.append(self.itos[index])
                if index == 0:
                    break
            print(''.join(output))

    def loss(self, test_word=None):
        """The idea is that we will take average negative log likelihood as our loss function.
        To find the loss function value:
        1. We will find the probability matrix for lookup table(lookup-table / sum(lookup-table)).
        2. We will then apply log function to particular char sequence and add the values. we are using
            log values because it is a monotonically increasing function which will make values easier to handle,
            and we can add these probabilities. because the probability of two sequences happening one after
            other is derived by multiplying those two values. since log multiplication is equal to summing of values.
        3. We will then apply negative sign to log value.
        4. We will then take the average value of negative log value as our main loss value.
        5. Lower the average negative log likelihood the better the prediction is."""

        probabilities = (self.lookup + 1).float()  # the reason for adding a numerical value is to smoothen out the
        # probabilities
        probabilities /= probabilities.sum(1, keepdims=True)  # this will return us the probability torch matrix.

        log_likelihood = 0.0
        count = 0
        if test_word:
            lines = [test_word]
        else:
            lines = self.lines
        for i in lines:
            chs = ['.'] + list(i) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                idx1 = self.stoi[ch1]
                idx2 = self.stoi[ch2]
                log_likelihood += torch.log(probabilities[idx1, idx2])
                count += 1
        negative_log_likelihood = -log_likelihood
        return f'Final average negative log likelihood: {negative_log_likelihood / count}'


class dataset():
    """This is the helper class for dataset. we will be using this in other classes."""

    def __init__(self, lines):
        self.lines = lines
        chars = sorted(list(set(''.join(self.lines))))
        self.stoi = {s: i + 1 for i, s in enumerate(chars)}
        self.stoi['.'] = 0
        self.itos = {s: i for i, s in self.stoi.items()}
        self.x = None
        self.y = None

    def encode(self, word):
        return self.stoi[word]

    def decode(self, integer):
        return self.itos[integer]

    def input_output(self):
        """This will give you the xs and ys of the dataset. This is useful for neural network class.
        Idea is that we will return two tensors that has encoded integers of first character and second one with
        encoded integers of second character. This will serve as input and output."""
        x, y = [], []
        for line in self.lines:
            line = ['.'] + list(line) + ['.']
            for ch1, ch2 in zip(line, line[1:]):
                x.append(self.stoi[ch1])
                y.append(self.stoi[ch2])
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        self.x = self.onehot()
        return self.y

    def onehot(self):
        return F.one_hot(self.x, num_classes=len(self.stoi)).float()


class neural_network(dataset):
    """In this class we will be using Neural network to predict the next character. We will be using dataset class as our
    helper function."""
    """The idea is that we will be taking first character as the input and second character as output and will be training on 
    the dataset. We will be using a simple network that has only one layer. This is a linear network(Y = X.W) and will be 
    applying softmax to get the final output layer. Then will be using multinomial function to get one output. In a way this 
    is a classification problem except we have a lot more classes 54 to be exact. """

    def __init__(self, lines):
        super().__init__(lines)
        self.gen = torch.Generator().manual_seed(9968)
        self.probabilities, self.weights, self.loss_value = self.load_model()
        self.learning_rate = 50
        self.reg = 0.5

    def forward(self):
        logits = self.x @ self.weights
        return logits

    def loss(self):
        """This is the implementation of average negative log likelihood loss"""
        loss = -(self.probabilities[torch.arange(self.x.shape[0]), self.y].log().mean()) + self.reg * (
            (self.weights ** 2).mean())
        return loss

    def backward(self):
        self.weights.grad = None
        self.loss_value.backward()

    def update_weights(self):
        self.weights.data += -self.learning_rate * self.weights.grad

    def train(self, iterations):
        bar = 10
        if iterations > 10:
            bar = iterations // 10
        progress = 0
        print("Progress: ", end='')
        self.input_output()
        for i in range(iterations):
            if i % bar == 0:
                print(f'{progress}#', end='')
                progress += 10
            logits = self.forward()
            self.softmax(logits)
            self.backward()
            self.update_weights()
        print(f'-------Final Loss: {self.loss_value}')
        self.save_model()

    def softmax(self, logits):
        counts = logits.exp()
        self.probabilities = counts / counts.sum(1, keepdims=True)
        self.loss_value = self.loss()

    def logits(self, xonehot):
        return xonehot @ self.weights

    def generate(self, number_of_lines):
        """This function generates the output based on the network. It does not return anything and directly prints
        into console"""
        for iterations in range(number_of_lines):
            index = 0
            output = []
            while True:
                xone_hot_out = F.one_hot(torch.tensor([index]), num_classes=54).float()
                logits_out = self.logits(xone_hot_out)
                counts_out = logits_out.exp()
                probs_out = counts_out / counts_out.sum(1, keepdims=True)
                index = torch.multinomial(probs_out, num_samples=1, replacement=True, generator=self.gen).item()
                output.append(self.decode(index))
                if index == 0:
                    break
            print(''.join(output))

    def load_model(self):
        try:
            open('Neural_logits.pkl', 'w')
        except FileNotFoundError:
            print("####Model not found.####")
        with open('Neural_logits.pkl', 'rb') as file:
            try:
                dict = pickle.load(file)
            except EOFError:
                dict = None
        if dict is None:
            weights = torch.randn((54, 54), generator=self.gen, requires_grad=True)
            loss_value = None
            probs = None
            return probs, weights, loss_value
        print(">>>>>>>>>>Model loaded<<<<<<<<<< \n")
        return dict['probs'], dict['weights'], dict['loss']

    def save_model(self):
        open('Neural_logits.pkl', 'w')
        dict = {}
        dict['probs'] = self.probabilities
        dict['weights'] = self.weights
        dict['loss'] = self.loss_value
        with open('Neural_logits.pkl', 'wb') as file:
            pickle.dump(dict, file)
            print("model is updated \n")

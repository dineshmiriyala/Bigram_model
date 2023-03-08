# imports
import pandas
import numpy
import torch
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


class bigram():
    """Bigram model is simply a lookup table of logits for the next character given the previous character."""

    def __init__(self, lines):
        """This has all the varibales for bigram model. This might not be the best appraoch to do this.
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
        This does not return anything but does operations on the existing self.lookup which is a
        torch. tensor with datatype as int 64"""
        for line in self.lines:
            line = ['.'] + list(line) + ['.']
            for ch1, ch2 in zip(line, line[1:]):
                idx1 = self.stoi[ch1]
                idx2 = self.stoi[ch2]
                self.lookup[idx1, idx2] += 1

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
        output = []
        for iterations in range(length):
            index = 0
            while True:
                probability = self.lookup[index].float()
                probability = probability / probability.sum()
                index = torch.multinomial(probability, num_samples=1, replacement=True, generator=generator).item()
                output.append(self.itos[index])
                if index == 0:
                    break

        return ''.join(output)


class bigram_with_NN():
    None


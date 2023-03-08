# imports
import torch
import warnings

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
        for iterations in range(length):
            output = []
            index = 0
            while True:
                probability = self.lookup[index].float()
                probability = probability / probability.sum()
                index = torch.multinomial(probability, num_samples=1, replacement=True, generator=generator).item()
                output.append(self.itos[index])
                print(''.join(output))
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


class bigram_with_NN():
    None

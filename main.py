"""This is the driver function for model"""

from model import bigram

lines = open('reddit_convos.txt' , 'r').read().splitlines()

bigram = bigram(lines)

print(bigram.generate(10))
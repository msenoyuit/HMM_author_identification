import pandas as pd
import numpy as np
from markovV2 import *
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
total_data = np.array(train_df)
test_split = int(.9*total_data.shape[0])
train_data = total_data[:test_split,:]
test_data = total_data[test_split:,:]
#print(data[0,-1])
#print(data[data[:,-1] == data[0,-1],1])
models = dict()
authors = list(set(total_data[:,-1]))
print(train_data.shape)
for author in authors:
    models[author] = train_markov_chain(train_data[train_data[:,-1] == author,1])

#testing
correct = 0
for row in test_data:
    best = ""
    author_score = []
    for author in authors:
        author_score.append(score_line(row[1],models[author]))
    index_max = author_score.index(max(author_score))
    author_max = authors[index_max]
    #print(author_score)
    #print(row[-1])
    #print(authors)
    #print(author_max == row[2])
    correct += int(author_max == row[2])
    #if(author_max != row[2]):
        #print(author_score)
        #print(row)

print(correct / test_data.shape[0])
print(correct)
print(test_data.shape)
#print(total_data.shape)

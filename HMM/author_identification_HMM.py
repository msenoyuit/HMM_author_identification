import pandas as pd
import numpy as np
import math
import markovModel

def loadData():
    train_df = np.array(pd.read_csv('../data/train.csv'))
    np.random.shuffle(train_df)
    test_df = np.array(pd.read_csv('../data/test.csv'))
    authors = list(set(train_df[:,-1]))
    return train_df, test_df, authors
    
def crossSplit(data, splitCount):
    #round length so it fits the split count, will lose at most splitCount-1 entries
    length = math.floor(data.shape[0]/splitCount)*splitCount
    splitLength = int(length / splitCount)
    #print(splitLength)
    #print(length)
    splitData = np.empty([splitCount], "object")
    for split in range(splitCount):
        splitStart= int(split/splitCount * length)
        splitEnd = splitStart + splitLength
        #print(splitStart, splitEnd)
        splitData[split] = data[splitStart:splitEnd,:]
        #test_split = int(.9*total_data.shape[0])
        #train_data = total_data[:test_split,:]
    return splitData

def setSplit(splitData, splitNumber):
    validateData = splitData[splitNumber];
    trainData = np.concatenate(np.delete(splitData, splitNumber));
    return validateData, trainData
    
            
def trainModel(authors, train_data, depth=1, char_level=False):
    models = dict()
    author_prob = dict()
    #print(train_data.shape)
    unique, counts = np.unique(train_data[:,2], return_counts=True)
    author_prob = dict(zip(unique, counts/train_data.shape[0]))
    #print(author_prob)
    for author in authors:
        models[author] = markovModel.train_markov_chain(train_data[train_data[:,-1] == author,1],depth=depth, char_level=char_level)
    return models, author_prob
#testing

def testAcc(authors, models, data, author_prob, depth):
    correct = 0
    for row in data:
        firstWord = row[1].split()[0]
        best = ""
        author_score = []
        for author in authors:
            prior = float(models[author]['prior'].get(firstWord , 1/len(models[author]['prior'])))
            author_score.append(markovModel.score_line(row[1],models[author]['transitions'], depth)*author_prob[author]*prior)
        index_max = author_score.index(max(author_score))
        author_max = authors[index_max]
        correct += int(author_max == row[2])
    return correct/len(data)


#print()
def markovModelTest(splitCount, depth, char_level):
    print("split count - ", str(splitCount), "\ndepth - ", str(depth), "\nchar_level - ", str(char_level))
    train_data, test_data, authors = loadData()
    if(splitCount == 0):
        splitCount = train_data.shape[0]
    splitData = crossSplit(train_data, splitCount)
    correct = 0
    for split in range(splitCount):
        print("split %d out of %d - %d" % (split,splitCount,split/splitCount*100), "%")
        validateData, trainData = setSplit(splitData,split)
        model, author_prob = trainModel(authors, trainData, depth, char_level)
        splitAcc = testAcc(authors,model,validateData, author_prob, depth)
        correct += splitAcc
        print("split acc - ", splitAcc)
    correct /= splitCount
    print("average acc- ", str(correct))
    return correct

markovModelTest(splitCount = 10, depth = 2, char_level=False)

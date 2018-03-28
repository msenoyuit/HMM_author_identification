import pandas as pd
import numpy as np
import math
import markovModel
import multiprocessing

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

    splitData = np.empty([splitCount], "object")
    for split in range(splitCount):
        splitStart= int(split/splitCount * length)
        splitEnd = splitStart + splitLength
        splitData[split] = data[splitStart:splitEnd,:]
    return splitData

def setSplit(splitData, splitNumber):
    validateData = splitData[splitNumber];
    trainData = np.concatenate(np.delete(splitData, splitNumber));
    return trainData, validateData
   
            
def trainModel(authors, train_data, depth=1, char_level=False):
    models = dict()
    author_prob = dict()
    unique, counts = np.unique(train_data[:,2], return_counts=True)
    author_prob = dict(zip(unique, counts/train_data.shape[0]))
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
 
 
def testSplit(args):
    trainData, validateData, authors, depth, char_level, split = args
    model, author_prob = trainModel(authors, trainData, depth, char_level)
    splitAcc = testAcc(authors,model,validateData, author_prob, depth)
    print("split -", str(split), "acc -", str(splitAcc) , str(trainData.shape), str(validateData.shape))
    return splitAcc


#print()
def markovModelTest(splitCount, depth, char_level):
    print("split count -", str(splitCount), "\ndepth -", str(depth), "\nchar_level -", str(char_level))
    train_data, test_data, authors = loadData()
    if(splitCount == 0):
        splitCount = train_data.shape[0]
    splitData = crossSplit(train_data, splitCount)
    correct = 0
    inList = []
    print(str(multiprocessing.cpu_count()), " cores found")
    for split in range(splitCount):
        trainData, validateData =  setSplit(splitData,split)
        inList.append((trainData, validateData, authors, depth, char_level, split))
        
     
    with multiprocessing.Pool(processes=None) as pool:
        correct = pool.map(testSplit, inList)
    
    correct = np.mean(correct)
    print("average acc- ", str(correct))
    return correct

if __name__ == '__main__':
    markovModelTest(splitCount = 10, depth =  1, char_level=False)

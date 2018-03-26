import glob
import os
import re
import h5py
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
import numpy as np
import pandas as pd

def import_files():
    train_data = np.array(pd.read_csv('../data/train.csv'))
    np.random.shuffle(train_data)
    test_data = np.array(pd.read_csv('../data/test.csv'))
    authors = list(set(train_data[:,-1]))
    return train_data, test_data, authors

    
    
def write_hdf5(dataDict, DataName, PPDataDir, verbose=False):
    if verbose:
        print("Saving processed data to"+PPDataDir+DataName+".hdf5"+" : ", end="", flush=True)
    if not os.path.isdir(PPDataDir):
        os.makedirs(PPDataDir)
    with h5py.File(PPDataDir+DataName+".hdf5", "a") as PPData:
        for name, data in dataDict.items():
            PPData[name] = data
    if verbose:
        print("Done")
        
        
        
def read_hdf5(DataName, PPDataDir, verbose=False):
    dataDict = dict()
    if verbose:
        print("Loading processed data form"+PPDataDir+DataName+".hdf5"+" : ", end="", flush=True)
    with h5py.File(PPDataDir+DataName+".hdf5", "r") as PPData:
        for name, data in PPData.items():
            dataDict[name] = data[:]
    if verbose:
        print("Done")
    return dataDict

    
    
def getListOfSequences(PPDataDir = "./processed_data/", verbose=False, lower=True, char_level=False, seq_size=1):
    savedDataName = re.sub(r'\W+', '', "author_identification_RNN")+'_cl'+str(int(char_level))+'_ss'+str(seq_size)
    if verbose:
        print("Loading and tokenizing data : ", end="", flush=True)
    train_data, test_data, authors = import_files()
    text_in = train_data[:,1]
    author_to_index = {}
    for author_index in range(len(authors)):
        author_to_index[authors[author_index]] = author_index
    labels_in = np.empty(train_data.shape[0])
    plays_token = text.Tokenizer(lower=lower, char_level=char_level)
    plays_token.fit_on_texts(text_in)
    if verbose:
        print("Done")
        
        
    if(os.path.exists(PPDataDir +savedDataName+".hdf5")):
        if verbose:
            print("Found preprocessed data.")
        dataDict = read_hdf5(savedDataName, PPDataDir, verbose=verbose)
        x_seq = dataDict['x_seq']
        y_seq = dataDict['y_seq']
            
            
    else:
        if verbose:
            print("No preprocessed data found.")
        text_sequences = plays_token.texts_to_sequences(text_in)
        sequences = list()
        if verbose:
            print("Splitting into seq : ", end="", flush=True)
        seq_authors = []
        for text_sequence, author in zip(text_sequences, train_data[:,2]):
            for i in range(1, len(text_sequence)):
                seq_authors.append(author_to_index[author])
                start = i-seq_size
                if(start<0):
                    start = 0
                seq = text_sequence[start:i]
                sequences.append(seq)

        y_seq = np.array(seq_authors)
        x_seq = sequence.pad_sequences(sequences, padding='post')
        dataDict = {'x_seq': x_seq, 'y_seq':y_seq}
        write_hdf5(dataDict, savedDataName, PPDataDir, verbose=verbose)
        
        
    return x_seq, y_seq, plays_token, authors


def sequence_to_text(seq, token, char_level=False):
    wordmap = {v: k for k,v in token.word_index.items()}
    script = ""
    for word in seq:
        script += wordmap[word]
        if not char_level: script += "1"
    return script



def genSequence(model, token, sample_len = 500, seq_size=50, char_level=False, verbose=False):
    if verbose:
        print("Generating text : ", end="", flush=True)

    seed_text = np.zeros([1,seq_size])
    prediction = ""
    for  _  in range(sample_len):
        predict = model.predict_classes(seed_text, verbose = 0)
        text = sequence_to_text([predict[0]], token, char_level)
        prediction += text
        seed_text = np.append(seed_text[:,1:],[predict], axis=1)

    if verbose:
        print("Done")
    print("-"*40 + 'Generated sequence' + '-'*40)
    return prediction


"""Functions for learning and sampling from word-based Markov models"""
import numpy as np
from collections import Counter
import gzip
import re


def train_markov_chain(sequences, depth=1, char_level=False, verbose=False):
    n = len(sequences)
    train_data = []
    all_tokens = set()
    
    # Flatten sequences into list separated by '\n'
    for sequence in sequences:
        # split strings by whitespace
        if char_level:
            tokens = list(re.sub("[^a-zA-Z ]+", "", sequence.lower()))
        else:
            tokens =  sequence.split()
        train_data += (tokens + ['\n'])  # '\n' is used to signal the end of a sequence
        all_tokens.update(tokens)

    #Create empty dicts/Counter for model and its parts
    model = dict()
    token_counter = Counter()
    prior = dict()

    #find and set the prior
    useNext = True
    for word in train_data:
        if(word == '\n'):
            useNext = True
        elif(useNext):
            token_counter[word] += 1
            useNext = False
    count = sum(token_counter.values())
    for first in token_counter:
        prior[first] = token_counter[first]/count
    model['prior'] = prior

    
    # compute transition probabilities
    transition_counter = dict()
    for word in range(0, len(train_data)-depth):
        add_transition_to_count(transition_counter, train_data[word:word+1+depth])
    #print(train_data)
    #print(transition_counter)
    #'''
    convert_transition_to_probabilities(transition_counter)

    model['transitions'] = transition_counter

    return model

    
    
def add_transition_to_count(word_dict, word_list):
    if(word_list[0] == '\n'):
        return
    if(len(word_list) == 1):
        word_dict[word_list[0]] += 1
        return
    if(word_list[0] not in word_dict):
        if(len(word_list) == 2):
            word_dict[word_list[0]] = Counter()
        else:
            word_dict[word_list[0]] = dict()
    add_transition_to_count(word_dict[word_list[0]], word_list[1:])
    
    
    
def convert_transition_to_probabilities(word_counter):
    words_list = list(word_counter.keys())
    if type(word_counter) is Counter:
        count = sum(list(word_counter.values()))
        for word in words_list:
            word_counter[word] /= count
    else:
        for word in words_list:
            convert_transition_to_probabilities(word_counter[word])
        
        
        
def score_seq(word_list, model):
    if word_list[0] not in model:
        return 1/(len(model)+1)
    if(len(word_list) == 1):
        return model[word_list[0]]
    else:
        return score_seq(word_list[1:], model[word_list[0]])
    
    
    
def score_line(line, model, depth):
    words = line.split()
    score = 1
    for word in range(len(words)-depth):
        score *= score_seq(words[word:word+1+depth], model)
    return score
    
    
    
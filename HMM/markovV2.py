"""Functions for learning and sampling from word-based Markov models"""
from __future__ import division
import numpy as np
from collections import Counter
import gzip
import re


def train_markov_chain(sequences):
    """Train a Markov chain model on a dataset of sequences.

    :param sequences: list of sequences of tokens. Each entry is a sequence to be modeled as sampled from a Markov
    chain, beginning with a token drawn from a prior distribution and ending with a stop token ('\n').
    :type sequences: list
    :return: Model dictionary describing the Markov transition probabilities ('transitions') and the prior
    probability ('prior') of the initial state.
    :rtype: dict
    """
    n = len(sequences)

    train_data = []

    all_tokens = set()

    for sequence in sequences:
        # split strings by whitespace
        #tokens = list(re.sub("[^a-zA-Z ]+", "", sequence.lower()))
        #print(tokens)
        #exit()
        train_data += (tokens + ['\n'])  # '\n' is used to signal the end of a sequence
        all_tokens.update(tokens)

    model = dict()

    token_counter = Counter()
    prior = dict()

    ##############################################################
    # 1 of 2
    # Insert your code here to fill prior with the correct value
    # The dictionary 'prior' should contain the probabilities of any
    # sequence starting with a word.
    # You can use token_counter to help compute the probabilities.
    # See https://docs.python.org/2/library/collections.html
    # for how to use Python Counter objects.
    ##############################################################
    #model, prior, token_counter, train_data, sequences
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
    
    ##################################################
    # End of 1 of 2
    ##################################################

    model['prior'] = prior

    # compute transition probabilities

    transition_counter = dict()
    transitions = dict()

    for token in all_tokens:
        transition_counter[token] = Counter()
        transitions[token] = dict()

    ####################################################################
    # 2 of 2
    # Insert your code here to fill transitions with the correct values
    # The dictionary 'transitions' should contain the conditional
    # probability of the next word given the current word. It should be
    # a dictionary of dictionaries.
    #
    # You can use the 'transition_counter' dictionary of Counter objects
    # to count the numbers you need to compute these probabilities.
    #
    # If a word has zero probability of being next, you should not store
    # a value in the dictionary to save computation. For example, if
    # 'abc' is never followed by '123', transitions['abc'] should not
    # contain an entry for '123', and calling transitions['abc']['123']
    # should trigger a KeyError.
    ####################################################################
    for word in range(0, len(train_data)-1):
        if(train_data[word] != '\n'):
            #print(train_data[word])
            transition_counter[train_data[word]][train_data[word+1]] += 1
    #print(train_data)
    #print(transition_counter)
    #'''
    for word in transition_counter:
        #print(transition_counter[word])
        count = sum(transition_counter[word].values())
        for nextWord in transition_counter[word]:
            #print(nextWord)
            transitions[word][nextWord] = transition_counter[word][nextWord]/count
    #print(transitions)
    #'''
    ##################################################
    # End of 2 of 2
    #################################################

    model['transitions'] = transitions

    return model


def score_line(line, model):
    words = line.split()
    score = 1
    for word in range(len(words)-1):
        if words[word] in model:
            if words[word+1] in model[words[word]]:
                score *= (model[words[word]][words[word+1]])
            else:
                score *= 1/len(model)
        else:
            score *= 1/len(model)
    return score
    
def sample_markov_chain(model, num_sequences, max_length=500):
    """Sample a series of sequences from a Markov chain.

    :param model: Model dictionary describing the Markov transition probabilities ('transitions') and the prior
    probability ('prior') of the initial state.
    :type model: dict
    :param num_sequences: Number of sequences to generate
    :type num_sequences: int
    :param max_length: Maximum length of each sequence. The Markov chain will generate a sequence until it samples a
    stop token or it reaches this maximum length.
    :type max_length: int
    :return: List of lists of tokens. Each sublist is a generated sequence from the Markov chain.
    :rtype: list
    """
    sequences = []

    ###################################################################
    # 1 of 1
    # Generate num_sequences sequences by sampling from the Markov model
    #
    # Start each sequence with a random token from the prior
    # distribution.
    #
    # End each sequence when you sample a '\n' token or when you reach
    # the maximum length 'max_length'.
    #
    # If no learned probabilities exist for the next token, end the
    # sequence. Otherwise, each next word should be randomly selected
    # from the conditional transition probability.
    #
    # Be careful about appending strings to a list of strings. If done
    # incorrectly, you can accidentally append each character to the list.
    ####################################################################
    for seqCount in range(num_sequences):
        sequence = []
        token = sample_token(model['prior'])
        sequence += token
        count = 1
        while(token != '\n' and count < max_length):
            count += 1
            token = sample_token(model['transitions'][token])
            sequence += token
        if(token != '\n'):
            sequence += '\n'
        sequences.append(sequence)
    #####################################################################
    # End of 1 of 1
    #####################################################################

    return sequences


def parse_text(filename):
    """Load a sequence of paragraphs from a text file into a list of
    paragraphs. This function treats a blank line as a paragraph break.

    :param filename: Name of text file encoded in utf-8 format.
    :type filename: string
    :return: List of sequences, where each sequence is the string of a
    paragraph.
    :rtype: list
    """
    sequences = []
    current_sequence = []

    # read each paragraph as a sequence
    with gzip.open(filename) as f:
        for line in f:
            line = line.decode('utf-8').strip()

            current_sequence += [line]

            if not line:
                # only if the line is empty, consider this a paragraph break
                next_sequence = " ".join(current_sequence).strip()
                if next_sequence:
                    sequences += [next_sequence]

                current_sequence = []

        # add last sequence
        sequences += [" ".join(current_sequence).strip()]

    return sequences


def sample_token(token_probs):
    """Sample from a dictionary representing a discrete multinomial distribution.

    :param token_probs: dictionary of token probabilities
    :type token_probs: dict
    :return: key of sampled entry
    :rtype: object
    """
    # convert dictionary to a list with a fixed order
    #print(token_probs.items())
    ordered_tokens, probabilities = zip(*token_probs.items())
    index = np.random.multinomial(1, probabilities).argmax()
    return ordered_tokens[index]

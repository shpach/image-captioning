import os
import pickle as pickle
import numpy as np

class WordTable():
    def __init__(self):
        self.word2vec = {}
        self.num_words = 0
        self.word_freq = []
        self.max_num_words = 0
        self.max_sentence_len = 0

    def build(self, sentences):
        word_count = {}
        for sent in sentences:
                for w in sent.lower().split(' '):
                    word_count[w] = word_count.get(w, 0) + 1
                    if w not in self.word2vec:
                        self.word2vec[w] = 0.01

    def load_gloves(self, dir):
        """ Using GloVe data for word embedding"""
        self.word2vec = {}
        glove_file = os.path.join(dir, 'glove.6B.'+str(self.dim_embed)+'d.txt')
        with open(glove_file) as f:
            for line in f:
                l = line.split()
                self.word2vec[l[0]] = [float(x)*0.05 for x in l[1:]]

    def save(self):
        """ Save the word table to pickle """

    def load(self):
        """ Load the word table from pickle """

import os
import pickle as pickle
import numpy as np

class WordTable():
    def __init__(self, vector_dim, save_file):
        self.word2vec = {}
        self.sentence_dictionary = {}
        self.num_words = 0
        self.word_freq = []
        self.max_num_words = 0
        self.max_sentence_len = 50
        self.dim_embed = vector_dim
        self.save_file = save_file
        self.word_count_threshold = 0

        self.idx2word = {}
        self.word2idx = {}


    def build(self, dir):

        token_file = dir + 'Flickr8k_text/Flickr8k.token.txt'
        with open(token_file) as f:
            for line in f:
                l = line.strip()
                sentence_split = l.split('\t', 1)
                self.sentence_dictionary[sentence_split[0]] = sentence_split[1]

        word_count = {}
        sentence_num = 0
        for key in self.sentence_dictionary:
            sentence = self.sentence_dictionary[key]
            for w in sentence.lower().split(' '):
                word_count[w] = word_count.get(w, 0) + 1
            sentence_num += 1
        

        vocab = []
        for key in self.sentence_dictionary:
            sentence = self.sentence_dictionary[key]
            for w in sentence.lower().split(' '):
                if word_count[w] >= self.word_count_threshold and not(w in vocab):
                    vocab.append(w)

        for idx, word in enumerate(vocab):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.num_words = len(self.word2idx)

    def load_gloves(self, dir):
        """ Using GloVe data for word embedding"""
        self.word2vec = {}
        glove_file = os.path.join(dir, 'glove.6B.'+str(self.dim_embed)+'d.txt')
        with open(glove_file, encoding="utf8") as f:
            for line in f:
                l = line.split()
                self.word2vec[l[0]] = [float(x) for x in l[1:]]


    def save(self):
        """ Save the word table to pickle """
        pickle.dump([self.word2vec, self.sentence_dictionary, self.word_freq, self.num_words, self.word2idx, self.idx2word], open(self.save_file, 'wb'), protocol=4)

    def load(self):
        """ Load the word table from pickle """
        self.word2vec, self.sentence_dictionary, self.word_freq, self.num_words, self.word2idx, self.idx2word = pickle.load(open(self.save_file, 'rb'))


import torch
import re
import unicodedata
import gensim
import numpy as np

class Lang:
    def __init__(self):
        self.word2index = {"[PAD]": 0}
        self.word2count = {}
        self.index2word = {0: "[PAD]"}
        self.n_words = 1
        self.word2vecMatrix = np.zeros(300) 

    def addSentence(self, sentence):
        words = sentence.split(' ')
        for word in words:
            self.addWord(word)
        return words

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    
    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def readFile(self, path):
        print("Reading lines...")
        with open(path) as f:
            lines = f.read().strip().split('\n')
        lines = [self.normalizeString(l) for l in lines]
        print("Finish reading!")
        return lines

    def tokenize(self, lines, maxLength):
        print("Tokenizing...")
        tokenized_texts = []
        for line in lines:
            tokenized_texts.append(self.addSentence(line))
        print("Finish tokenizeing!")
        indexed_texts = []
        for line in tokenized_texts:
            indexed_words = []
            for word in line:
                indexed_words.append(self.word2index[word])
            if len(indexed_words) > maxLength:
                indexed_words = indexed_words[:maxLength]
            elif len(indexed_words) < maxLength:
                indexed_words += [0]*(maxLength-len(indexed_words))
            indexed_texts.append(indexed_words)
        return indexed_texts  

    def loadWord2Vec(self, path):
        print("Loading word2vec...")
        model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        word2vec = {}
        word2vec["[PAD]"] = np.zeros(300, dtype='float32')
        for word in self.index2word.values():
            if word in model:
                word2vec[word] = model[word]
            else:
                word2vec[word] = np.random.uniform(-0.25, 0.25, 300)
        vocabSize = len(self.index2word)
        self.word2vecMatrix = np.zeros(shape=(vocabSize, 300), dtype='float32')
        for i in range(0, vocabSize):
            self.word2vecMatrix[i] = word2vec[self.index2word[i]]
        print("Finish Loading!")
        return self.word2vecMatrix
        

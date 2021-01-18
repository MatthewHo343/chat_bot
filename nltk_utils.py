import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

# tokenizes the sentence (splitting the text)
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# stems the word (reduce word to base)
def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype = np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1
    return bag

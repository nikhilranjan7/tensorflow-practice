'''
First a lexicon(array) of all defined words are give,
lexicon = [chair, table, spoon, television]
Now suppose a new sentence is given,
I pulled the chair up to the table
Now np.zeros(len(lexicon)) array is initialized, with one-to-one count
correspondence to lexicon list [0 0 0 0]
After going through the sentence lexicon_count will convert to [1 1 0 0]
'''

import nltk
from nltk.tokenize import word_tokenize     # take the sentence and separate the words in a list
from nltk.stem import WordNetLemmatizer     # Running, run, ran would be converted to a same word
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

# If MemoryError is given, RAM is full

def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    # w_counts will looks like {'the':52521,'and':25242}
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print("Length of Lexicon: ", len(lexicon))
    return l2 # l2 is our final filtered lexicon

def sample_handling(sample, lexicon, classification):
    featureset = []
    ''' [
    [[0 1 0 0 1 0 1], [1 0]],
    ]
    featureset will look like above
    '''
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset

def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('data/pos.txt', lexicon, [1,0])
    features += sample_handling('data/neg.txt', lexicon, [0,1])
    random.shuffle(features)
    features = np.array(features)
    testing_size = int(test_size*len(features))
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])

    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x, train_y, test_x, test_y

if __name__ == "__main__":
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('data/pos.txt', 'data/neg.txt')
    with open('sentiment_set.pickle','wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)

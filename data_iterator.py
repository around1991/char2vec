from __future__ import print_function

from collections import defaultdict
import numpy
numpy.random.seed(45)
import random
random.seed(45)

import pdb

from scipy.stats import spearmanr

import keras.backend as K
from keras import initializations, regularizers
from keras.callbacks import Callback

import theano.tensor as T


def word_to_indices(word):
    """Converts a word to a sequence of character IDs"""
    indices = [ord(letter) - ord('a') + 1 for letter in word]
    assert(max(indices) < 27)
    return indices


def infinite_cycle(iterator):
    while True:
        for item in iterator:
            yield item


class FileIterator(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename, 'r') as f:
            for line in f:
                yield line.split()


class Word2VecIterator(object):
    """Reads a corpus, and spits out pairs to feed to a model"""

    def __init__(self, corpus, n_neg, batch_size, window, min_count,
                 model=None):
        self.corpus = corpus
        self.n_neg = n_neg
        self.batch_size = batch_size
        self.window = window
        self.min_buflen = 100000
        self.max_buflen = 2 * self.min_buflen

        self.min_count = min_count

        self.neg_table_size = 50000000

        self._length = 0

        self.word_counts = defaultdict(int)

        if not model:
            self.get_counts()
            self.process_vocab(self.min_count)
            self.make_neg_table()

        else:
            self.word2index = {}
            for word in model.vocab:
                self.word_counts[word] = model.vocab[word].count
                self.word2index[word] = model.vocab[word].index

            self.word_index = sorted(list(self.word2index.keys()),
                                     key=lambda x: model.vocab[x].index)
            self.make_neg_table()

    def get_counts(self):
        for line in self.corpus:
            for word in line:
                self.word_counts[word] += 1

    def process_vocab(self, min_count):
        self.word_counts = {word: self.word_counts[word] for word in self.word_counts
                            if self.word_counts[word] >= min_count}
        self.word_index = self.word_counts.keys()
        self.word2index = {self.word_index[i]: i for i in range(len(self.word_index))}

    def make_neg_table(self):
        # make negative sampling table
        vocab_size = len(self.word_counts)
        self.neg_table = numpy.zeros(self.neg_table_size, dtype=numpy.uint32)
        train_words_pow = float(sum(self.word_counts[word]**0.75 for word in self.word_counts))
        cumulative = 0
        table_idx = 0
        for index in range(vocab_size):
            cumulative += self.word_counts[self.word_index[index]]**0.75 / train_words_pow
            while table_idx < int(cumulative * self.neg_table_size):
                self.neg_table[table_idx] = index
                table_idx += 1

        while table_idx < self.neg_table_size:
            self.neg_table[table_idx] = vocab_size - 1
            table_idx += 1
        assert self.neg_table[-1] == vocab_size - 1

    def __len__(self):
        # account for negative samples

        if self._length == 0:
            for line in self.corpus:
                # filter the line of OOV words and subsample common words
                line = [word for word in line if word in self.word_counts]
                for pos, word in enumerate(line):
                    start = max(0, pos - self.window)
                    for pos2, word2 in enumerate(line[start:(pos + self.window)], start):
                        if pos2 != pos:
                            # account for negative samples
                            self._length += (1 + self.n_neg)

        return self._length

    def __iter__(self):
        buffer = {}
        current_key = 0

        def make_batch(batch_size):
            content_forward = []
            content_backward = []
#            contents = []
            contexts = []
            labels = []
            weights = []
            keys = buffer.keys()
            for i, key in enumerate(keys):
                if i < batch_size:
                    item = buffer.pop(key)
                    content_forward.append([27] + item[0])
                    content_backward.append(item[0] + [28])
    #                contents.append(item[0])
                    contexts.append(item[1])
                    labels.append(item[2])
                    weights.append(item[3])
                else:
                    break
            max_content_length = max(len(content)
                                     for content in content_forward)
            l1 = numpy.zeros((batch_size, max_content_length), dtype=numpy.int32)
            l2 = numpy.zeros((batch_size, max_content_length), dtype=numpy.int32)
            for i, (forward, backward) in enumerate(zip(content_forward,
                                                        content_backward)):
                l1[i, :len(forward)] = forward
                l2[i, :len(backward)] = backward
#            l = numpy.array(contents, dtype=numpy.int32)
            r = numpy.array(contexts, dtype=numpy.int32)
            labels = numpy.array(labels)
            weights = numpy.array(weights)
            return ([l1, l2, r],  # these are the inputs
                    labels,  # these are the targets
                    weights)  # these are the sample weights

        for line in self.corpus:
            # filter the line of OOV words and subsample common words
            line = [word for word in line
                    if word in self.word_counts]
            for pos, word in enumerate(line):
                word_chars = word_to_indices(word)
                start = max(0, pos - self.window)
                for pos2, word2 in enumerate(line[start:(pos + self.window)], start):
                    if pos2 != pos:
                        # add both words to the batch with label 1
                        buffer[current_key] = (word_chars, [self.word2index[word2]], 1., 1.)
                        current_key += 1

                        neg_samples = []
                        # add n_neg negative samples
                        while len(neg_samples) < self.n_neg:
                            word2 = self.neg_table[numpy.random.randint(0, self.neg_table_size)]
                            if word2 not in [self.word2index[word]] + neg_samples:
                                neg_samples.append(word2)
                                buffer[current_key] = (word_chars, [word2], 0., 1.)
                                current_key += 1

            if len(buffer) > self.max_buflen:
                while len(buffer) > self.min_buflen:
                    # yield some batches to clear space in the buffer
                    yield make_batch(min(self.batch_size, len(buffer)))

        # empty the buffer when we've reached the end of the corpus
        while len(buffer) > 0:
            yield make_batch(min(self.batch_size, len(buffer)))


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.seen = 0
        self.total_seen = []

    def on_batch_begin(self, batch, logs={}):
        self.seen += logs.get('size')

    def on_batch_end(self, batch, logs={}):
        self.total_seen.append(self.seen)
        self.losses.append(logs.get('loss'))


class MENEvaluator(Callback):
    def __init__(self, model):
        self.wordseg_model = model

    def on_train_begin(self, logs={}):
        self.similarities = {}
        # load the MEN dev set and strip pos tags
        with open('/local/scratch/kc391/word_similarity_data/MEN/MEN_dataset_lemma_form.dev', 'r') as f:
            for line in f:
                line = line.split()
                word_pair = (line[0][:-2], line[1][:-2])
                sim = float(line[2])
                self.similarities[word_pair] = sim

    def on_epoch_end(self, epoch, logs={}):
        predicted_similarities_all = []
        gold_similarities_all = []
        predicted_similarities_res = []
        gold_similarities_res = []
        for word_pair in self.similarities.keys():
            word1_vec = self.wordseg_model.predict(word_pair[0])[0][0]
            word2_vec = self.wordseg_model.predict(word_pair[1])[0][0]

            sim = (numpy.dot(word1_vec, word2_vec.T) /
                   (numpy.linalg.norm(word1_vec) * numpy.linalg.norm(word2_vec)))

            predicted_similarities_all.append(sim)
            gold_similarities_all.append(self.similarities[word_pair])

            if all(word in self.wordseg_model.word2vec_model
                   for word in word_pair):
                predicted_similarities_res.append(sim)
                gold_similarities_res.append(self.similarities[word_pair])

        r_all = spearmanr(predicted_similarities_all, gold_similarities_all)
        r_res = spearmanr(predicted_similarities_res, gold_similarities_res)

        print("r all: {0:.4f}; r res: {1:.4f}".format(r_all[0], r_res[0]))

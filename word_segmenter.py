import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy
from scipy.spatial.distance import cosine

from keras.engine import Model, merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import (Flatten, Dense, Activation, Lambda)
from keras.layers import Input
from keras.layers.wrappers import TimeDistributed
import keras.backend as K


from data_iterator import (Word2VecIterator, MENEvaluator,
                           infinite_cycle, LossHistory, word_to_indices)


class WordSegmenter(object):
    def __init__(self, word2vec_model):
        self.word_vectors = None
        self.word2vec_model = word2vec_model

    def build_iterator(self, corpus, n_neg, batch_size, window, min_count):
        self.iterator = Word2VecIterator(corpus, n_neg, batch_size, window,
                                         min_count, self.word2vec_model)

    def build(self, learn_context_weights=True):
        content_forward = Input(shape=(None,), dtype='int32',
                                name='content_forward')
        content_backward = Input(shape=(None,), dtype='int32',
                                 name='content_backward')
        context = Input(shape=(1,), dtype='int32', name='context')

        if learn_context_weights:
            context_weights = None
        else:
            context_weights = [self.word2vec_model.syn1neg]
        context_embedding = Embedding(input_dim=len(self.iterator.word_index),
                                      output_dim=256, input_length=1,
                                      weights=context_weights)
        if not learn_context_weights:
            context_embedding.trainable = False
        context_flat = Flatten()(context_embedding(context))

        char_embedding = Embedding(
            input_dim=29, output_dim=64, mask_zero=True)

        embed_forward = char_embedding(content_forward)
        embed_backward = char_embedding(content_backward)

        rnn_forward = LSTM(output_dim=256, return_sequences=True,
                           activation='tanh')(embed_forward)
        backwards_lstm = LSTM(output_dim=256, return_sequences=True,
                              activation='tanh', go_backwards=True)

        def reverse_tensor(inputs, mask):
            return inputs[:, ::-1, :]

        def reverse_tensor_shape(input_shapes):
            return input_shapes

        reverse = Lambda(reverse_tensor, output_shape=reverse_tensor_shape)
        reverse.supports_masking = True

        rnn_backward = reverse(backwards_lstm(embed_backward))

        rnn_bidi = TimeDistributed(Dense(output_dim=256))(
            merge([rnn_forward, rnn_backward], mode='concat'))

        attention_1 = TimeDistributed(Dense(output_dim=256,
                                            activation='tanh',
                                            bias=False))(rnn_bidi)
        attention_2 = TimeDistributed(Dense(output_dim=1,
                                            activity_regularizer='activity_l2',
                                            bias=False))(attention_1)

        def attn_merge(inputs, mask):
            vectors = inputs[0]
            logits = inputs[1]
            # Flatten the logits and take a softmax
            logits = K.squeeze(logits, axis=2)
            pre_softmax = K.switch(mask[0], logits, -numpy.inf)
            weights = K.expand_dims(K.softmax(pre_softmax))
            return K.sum(vectors * weights, axis=1)

        def attn_merge_shape(input_shapes):
            return(input_shapes[0][0], input_shapes[0][2])

        attn = Lambda(attn_merge, output_shape=attn_merge_shape)
        attn.supports_masking = True
        attn.compute_mask = lambda inputs, mask: None
        content_flat = attn([rnn_bidi, attention_2])

        output = Activation('sigmoid', name='output')(
            merge([content_flat, context_flat], mode='dot',
                  dot_axes=(1, 1)))


        model = Model(input=[content_forward, content_backward, context],
                      output=output)
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        inputs = [content_forward, content_backward]

        self._predict = K.function(inputs, content_flat)
        self._attention = K.function(inputs, K.squeeze(attention_2, axis=2))
        self.model = model

    def train(self, epochs, model_name):
        history = LossHistory()
        evaluation = MENEvaluator(self)

        self.model.fit_generator(iter(infinite_cycle(self.iterator)),
                                 len(self.iterator),
                                 epochs,
                                 callbacks=[history, evaluation])

        plt.figure()
        plt.plot(history.total_seen, history.losses)
        plt.ylim((0.0, 0.5))
        plt.savefig(model_name + '.png')

    def predict(self, word):
        indices = word_to_indices(word)

        forward = numpy.array([[27] + indices])
        backward = numpy.array([indices + [28]])

        embedding = self._predict([forward, backward])
        attention = self._attention([forward, backward])

        return embedding, attention

    def most_similar(self, word, n=10, context=False, cutoff=None):
        if not self.word_vectors:
            self.word_vectors = [self.predict(x)[0][0] for x in self.iterator.word_index]

        distances = numpy.array(
            [cosine(y, self.word_vectors[self.iterator.word2index[word]])
             for y in self.word_vectors])

        sorted_args = numpy.argsort(distances)
        if cutoff:
            sorted_args = [arg for arg in sorted_args
                           if self.iterator.word_count[self.iterator.word_index[y]] >= cutoff]
        words = [self.iterator.word_index[y] for y in sorted_args]

        return zip(distances[sorted_args][:n], words[:n])

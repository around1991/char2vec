# char2vec
Implementation of char2vec model from http://www.aclweb.org/anthology/W/W16/W16-1603.pdf

These two files implement the Char2Vec model from "A Joint Model for Word Embedding and Word Morphology" by Kris Cao and
Marek Rei, 2016. To get started:

* Create a corpus
* Create a WordSegmenter object (this is the model)
* Create a data iterator
* Build and train the model
* ???
* PROFIT

## Creating a corpus
A corpus is just any iterator which spits out a list of tokens which are in the same context. The Gensim Text8Corpus class is
an example of this: it takes the path to a local version of the Text8 corpus and spits out 1000 word chunks of the corpus.

Alternatively, if you have a file where each line is a sentence, then the FileIterator class in data_iterator.py does the 
same thing.

## Creating a WordSegmenter object
The WordSegmenter object is the implementation of the model. The `word2vec_model` argument of the constructor allows you to
pass a pretrained Gensim Word2Vec model, to speed up getting the word indexes for the data iterator and to load pre-trained
context vectors to help train the model. The `build_iterator` method builds a data iterator to train the model with: read on
for more information.

## Creating a data iterator
Call `build_iterator` with your preferred hyperparameters to construct a data iterator for training. For our values, read 
the paper.

## Training the model
Call the `build` method to construct the model. If you passed in a pre-trained word2vec model in the constructor, you can 
also pass in `learn_context_weights=False` in the arguments to use the pre-trained word2vec context weights.

Now call the `train` method to train your model for the specified number of epochs and save the weights and graph of loss 
with the specified name.

## ???
Watch the numbers go down.

## PROFIT
* The `predict` method predicts an embedding and a segmentation for any word you give it. Play around, even with OOV words!
* The `most_similar` method gives you the most similar words to your target word in the training vocabulary.

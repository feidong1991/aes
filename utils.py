# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2016-10-17 14:31:13
# @Last Modified by:   feidong1991
# @Last Modified time: 2017-02-12 20:46:40
import gzip
import logging
import sys
from gensim.models.word2vec import Word2Vec
import theano
import numpy as np
# import matplotlib.pyplot as plt


def get_logger(name, level=logging.INFO, handler=sys.stdout,
        formatter='%(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def padding_sentence_sequences(index_sequences, scores, max_sentnum, max_sentlen, post_padding=True):

    X = np.empty([len(index_sequences), max_sentnum, max_sentlen], dtype=np.int32)
    Y = np.empty([len(index_sequences), 1], dtype=np.float32)
    mask = np.zeros([len(index_sequences), max_sentnum, max_sentlen], dtype=theano.config.floatX)

    for i in range(len(index_sequences)):
        sequence_ids = index_sequences[i]
        num = len(sequence_ids)

        for j in xrange(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            # X_len[i] = length
            for k in xrange(length):
                wid = word_ids[k]
                # print wid
                X[i, j, k] = wid

            # Zero out X after the end of the sequence
            X[i, j, length:] = 0
            # Make the mask for this sample 1 within the range of length
            mask[i, j, :length] = 1

        X[i, num:, :] = 0
        Y[i] = scores[i]
    return X, Y, mask


def padding_sequences(word_indices, char_indices, scores, max_sentnum, max_sentlen, maxcharlen, post_padding=True):
    # support char features
    X = np.empty([len(word_indices), max_sentnum, max_sentlen], dtype=np.int32)
    Y = np.empty([len(word_indices), 1], dtype=np.float32)
    mask = np.zeros([len(word_indices), max_sentnum, max_sentlen], dtype=theano.config.floatX)

    char_X = np.empty([len(char_indices), max_sentnum, max_sentlen, maxcharlen], dtype=np.int32)

    for i in range(len(word_indices)):
        sequence_ids = word_indices[i]
        num = len(sequence_ids)

        for j in xrange(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            # X_len[i] = length
            for k in xrange(length):
                wid = word_ids[k]
                # print wid
                X[i, j, k] = wid

            # Zero out X after the end of the sequence
            X[i, j, length:] = 0
            # Make the mask for this sample 1 within the range of length
            mask[i, j, :length] = 1

        X[i, num:, :] = 0
        Y[i] = scores[i]

    for i in xrange(len(char_indices)):
        sequence_ids = char_indices[i]
        num = len(sequence_ids)
        for j in xrange(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            for k in xrange(length):
                wid = word_ids[k]
                charlen = len(wid)
                for l in xrange(charlen):
                    cid = wid[l]
                    char_X[i, j, k, l] = cid
                char_X[i, j, k, charlen:] = 0
            char_X[i, j, length:, :] = 0
        char_X[i, num:, :] = 0
    return X, char_X, Y, mask


def load_word_embedding_dict(embedding, embedding_path, word_alphabet, logger, embedd_dim=100):
    """
    load word embeddings from file
    :param embedding:
    :param embedding_path:
    :param logger:
    :return: embedding dict, embedding dimention, caseless
    """
    if embedding == 'word2vec':
        # loading word2vec
        logger.info("Loading word2vec ...")
        word2vec = Word2Vec.load_word2vec_format(embedding_path, binary=False, unicode_errors='ignore')
        embedd_dim = word2vec.vector_size
        return word2vec, embedd_dim, False
    elif embedding == 'glove':
        # loading GloVe
        logger.info("Loading GloVe ...")
        embedd_dim = -1
        embedd_dict = dict()
        with gzip.open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=theano.config.floatX)
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
        return embedd_dict, embedd_dim, True
    elif embedding == 'senna':
        # loading Senna
        logger.info("Loading Senna ...")
        embedd_dim = -1
        embedd_dict = dict()
        with gzip.open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=theano.config.floatX)
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
        return embedd_dict, embedd_dim, True
    # elif embedding == 'random':
    #     # loading random embedding table
    #     logger.info("Loading Random ...")
    #     embedd_dict = dict()
    #     words = word_alphabet.get_content()
    #     scale = np.sqrt(3.0 / embedd_dim)
    #     # print words, len(words)
    #     for word in words:
    #         embedd_dict[word] = np.random.uniform(-scale, scale, [1, embedd_dim])
    #     return embedd_dict, embedd_dim, False
    else:
        raise ValueError("embedding should choose from [word2vec, senna]")


def build_embedd_table(word_alphabet, embedd_dict, embedd_dim, logger, caseless):
    scale = np.sqrt(3.0 / embedd_dim)
    embedd_table = np.empty([len(word_alphabet), embedd_dim], dtype=theano.config.floatX)
    embedd_table[0, :] = np.zeros([1, embedd_dim])
    oov_num = 0
    for word, index in word_alphabet.iteritems():
        ww = word.lower() if caseless else word
        # show oov ratio
        if ww in embedd_dict:
            embedd = embedd_dict[ww]
        else:
            embedd = np.random.uniform(-scale, scale, [1, embedd_dim])
            oov_num += 1
        embedd_table[index, :] = embedd
    oov_ratio = float(oov_num)/(len(word_alphabet)-1)
    logger.info("OOV number =%s, OOV ratio = %f" % (oov_num, oov_ratio))
    return embedd_table


def rescale_tointscore(scaled_scores, set_ids):
    '''
    rescale scaled scores range[0,1] to original integer scores based on  their set_ids
    :param scaled_scores: list of scaled scores range [0,1] of essays
    :param set_ids: list of corresponding set IDs of essays, integer from 1 to 8
    '''
    # print type(scaled_scores)
    # print scaled_scores[0:100]
    if isinstance(set_ids, int):
        prompt_id = set_ids
        set_ids = np.ones(scaled_scores.shape[0],) * prompt_id
    assert scaled_scores.shape[0] == len(set_ids)
    int_scores = np.zeros((scaled_scores.shape[0], 1))
    for k, i in enumerate(set_ids):
        assert i in xrange(1, 9)
        # TODO
        if i == 1:
            minscore = 2
            maxscore = 12
        elif i == 2:
            minscore = 1
            maxscore = 6
        elif i in [3, 4]:
            minscore = 0
            maxscore = 3
        elif i in [5, 6]:
            minscore = 0
            maxscore = 4
        elif i == 7:
            minscore = 0
            maxscore = 30
        elif i == 8:
            minscore = 0
            maxscore = 60
        else:
            print "Set ID error"
        # minscore = 0
        # maxscore = 60

        int_scores[k] = scaled_scores[k]*(maxscore-minscore) + minscore

    return np.around(int_scores).astype(int)


def domain_specific_rescale(y_true, y_pred, set_ids):
    '''
    rescaled scores to original integer scores based on their set ids
    and partition the score list based on its specific prompot
    return 8-prompt int score list for y_true and y_pred respectively
    :param y_true: true score list, contains all 8 prompts
    :param y_pred: pred score list, also contains 8 prompts
    :param set_ids: list that indicates the set/prompt id for each essay
    '''
    # prompts_truescores = []
    # prompts_predscores = []
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    y1_true, y1_pred = [], []
    y2_true, y2_pred = [], []
    y3_true, y3_pred = [], []
    y4_true, y4_pred = [], []
    y5_true, y5_pred = [], []
    y6_true, y6_pred = [], []
    y7_true, y7_pred = [], []
    y8_true, y8_pred = [], []

    for k, i in enumerate(set_ids):
        assert i in xrange(1, 9)
        if i == 1:
            minscore = 2
            maxscore = 12
            y1_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y1_pred.append(y_pred[k]*(maxscore-minscore) + minscore)
        elif i == 2:
            minscore = 1
            maxscore = 6
            y2_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y2_pred.append(y_pred[k]*(maxscore-minscore) + minscore)
        elif i == 3:
            minscore = 0
            maxscore = 3
            y3_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y3_pred.append(y_pred[k]*(maxscore-minscore) + minscore)
        elif i == 4:
            minscore = 0
            maxscore = 3
            y4_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y4_pred.append(y_pred[k]*(maxscore-minscore) + minscore)

        elif i == 5:
            minscore = 0
            maxscore = 4
            y5_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y5_pred.append(y_pred[k]*(maxscore-minscore) + minscore)
        elif i == 6:
            minscore = 0
            maxscore = 4
            y6_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y6_pred.append(y_pred[k]*(maxscore-minscore) + minscore)

        elif i == 7:
            minscore = 0
            maxscore = 30
            y7_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y7_pred.append(y_pred[k]*(maxscore-minscore) + minscore)

        elif i == 8:
            minscore = 0
            maxscore = 60
            y8_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y8_pred.append(y_pred[k]*(maxscore-minscore) + minscore)

        else:
            print "Set ID error"
    prompts_truescores = [np.around(y1_true), np.around(y2_true), np.around(y3_true), np.around(y4_true), \
                            np.around(y5_true), np.around(y6_true), np.around(y7_true), np.around(y8_true)]
    prompts_predscores = [np.around(y1_pred), np.around(y2_pred), np.around(y3_pred), np.around(y4_pred), \
                            np.around(y5_pred), np.around(y6_pred), np.around(y7_pred), np.around(y8_pred)]

    return prompts_truescores, prompts_predscores
# def plot_convergence(train_stats, dev_stats, test_stats, metric_type='mse'):
#     '''
#     Plot convergence curve of training process
#     :param train_stats: list of train metrics at each epoch
#     :param dev_stats: list of dev metrics at each epoch
#     :param test_stas: list of test metrics at each epoch
#     '''
#     num_epochs = len(train_stats)
#     x = xrange(1, num_epochs+1)

#     plt.plot(x, train_stats)
#     plt.plot(x, dev_stats)
#     plt.plot(x, test_stats)
#     plt.legend(['train', 'dev', 'test'], loc='upper right')
#     plt.xlabel('num of epochs')
#     if metric_type == 'kappa':
#         y_label = 'Kappa value'
#     else:
#         y_label = 'Mean square error'
#     plt.ylabel('%s' % y_label)
#     plt.show()

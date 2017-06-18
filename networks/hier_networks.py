# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-10 11:40:53
# @Last Modified by:   feidong1991
# @Last Modified time: 2017-06-18 16:08:08

from keras.models import *
from keras.optimizers import *
from keras.layers.core import *
from keras.layers import Input, Embedding, LSTM, GRU, Dense, merge
from keras.layers import TimeDistributed

from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.regularizers import l2

from softattention import Attention
from zeromasking import ZeroMaskedEntries
from utils import get_logger
import time

logger = get_logger("Build model")

"""
Hierarchical networks, the following function contains several models:
(1)build_hcnn_model: hierarchical CNN model
(2)build_hrcnn_model: hierarchical Recurrent CNN model, LSTM stack over CNN,
 it supports two pooling methods
    (a): Mean-over-time pooling
    (b): attention pooling
"""


def build_hcnn_model(opts, vocab_size=0, maxnum=50, maxlen=50, embedd_dim=50, embedding_weights=None, verbose=False):

    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, filter2_len = %s, drop rate = %s, l2 = %s" % (N, L, embedd_dim,
        opts.nbfilters, opts.filter1_len, opts.filter2_len, opts.dropout, opts.l2_value))

    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, name='x')(word_input)
    drop_x = Dropout(opts.dropout, name='drop_x')(x)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    z = TimeDistributed(Convolution1D(opts.nbfilters, opts.filter1_len, border_mode='valid'), name='z')(resh_W)

    avg_z = TimeDistributed(AveragePooling1D(pool_length=L-opts.filter1_len+1), name='avg_z')(z)	# shape= (N, 1, nbfilters)

    resh_z = Reshape((N, opts.nbfilters), name='resh_z')(avg_z)		# shape(N, nbfilters)

    hz = Convolution1D(opts.nbfilters, opts.filter2_len, border_mode='valid', name='hz')(resh_z)
    # avg_h = MeanOverTime(mask_zero=True, name='avg_h')(hz)

    avg_hz = GlobalAveragePooling1D(name='avg_hz')(hz)
    y = Dense(output_dim=1, activation='sigmoid', name='output')(avg_hz)

    model = Model(input=word_input, output=y)

    if verbose:
        model.summary()

    start_time = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f s" % total_time)

    return model


def build_hrcnn_model(opts, vocab_size=0, char_vocabsize=0, maxnum=50, maxlen=50, maxcharlen=20, embedd_dim=50, embedding_weights=None, verbose=False, init_mean_value=None):
    # LSTM stacked over CNN based on sentence level
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s" % (N, L, embedd_dim,
        opts.nbfilters, opts.filter1_len, opts.dropout))

    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    # add char-based CNN, concatenating with word embedding to compose word representation
    if opts.use_char:
        char_input = Input(shape=(N*L*maxcharlen,), dtype='int32', name='char_input')
        xc = Embedding(output_dim=opts.char_embedd_dim, input_dim=char_vocabsize, input_length=N*L*maxcharlen, mask_zero=True, name='xc')(char_input)
        xc_masked = ZeroMaskedEntries(name='xc_masked')(xc)
        drop_xc = Dropout(opts.dropout, name='drop_xc')(xc_masked)
        res_xc = Reshape((N*L, maxcharlen, opts.char_embedd_dim), name='res_xc')(drop_xc)
        cnn_xc = TimeDistributed(Convolution1D(opts.char_nbfilters, opts.filter2_len, border_mode='valid'), name='cnn_xc')(res_xc)
        max_xc = TimeDistributed(GlobalMaxPooling1D(), name='avg_xc')(cnn_xc)
        res_xc2 = Reshape((N, L, opts.char_nbfilters), name='res_xc2')(max_xc)

        w_repr = merge([resh_W, res_xc2], mode='concat', name='w_repr')
        zcnn = TimeDistributed(Convolution1D(opts.nbfilters, opts.filter1_len, border_mode='valid'), name='zcnn')(w_repr)
    else:
        zcnn = TimeDistributed(Convolution1D(opts.nbfilters, opts.filter1_len, border_mode='valid'), name='zcnn')(resh_W)

    # pooling mode
    if opts.mode == 'mot':
        logger.info("Use mean-over-time pooling on sentence")
        avg_zcnn = TimeDistributed(GlobalAveragePooling1D(), name='avg_zcnn')(zcnn)
    elif opts.mode == 'att':
        logger.info('Use attention-pooling on sentence')
        avg_zcnn = TimeDistributed(Attention(), name='avg_zcnn')(zcnn)
    elif opts.mode == 'merged':
        logger.info('Use mean-over-time and attention-pooling together on sentence')
        avg_zcnn1 = TimeDistributed(GlobalAveragePooling1D(), name='avg_zcnn1')(zcnn)
        avg_zcnn2 = TimeDistributed(Attention(), name='avg_zcnn2')(zcnn)
        avg_zcnn = merge([avg_zcnn1, avg_zcnn2], mode='concat', name='avg_zcnn')
    else:
        raise NotImplementedError
    hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)

    if opts.mode == 'mot':
        logger.info('Use mean-over-time pooling on text')
        avg_hz_lstm = GlobalAveragePooling1D(name='avg_hz_lstm')(hz_lstm)
    elif opts.mode == 'att':
        logger.info('Use attention-pooling on text')
        avg_hz_lstm = Attention(name='avg_hz_lstm')(hz_lstm)
    elif opts.mode == 'merged':
        logger.info('Use mean-over-time and attention-pooling together on text')
        avg_hz_lstm1 = GlobalAveragePooling1D(name='avg_hz_lstm1')(hz_lstm)
        avg_hz_lstm2 = Attention(name='avg_hz_lstm2')(hz_lstm)
        avg_hz_lstm = merge([avg_hz_lstm1, avg_hz_lstm2], mode='concat', name='avg_hz_lstm')
    else:
        raise NotImplementedError
    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(output_dim=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_lstm)
    else:
        y = Dense(output_dim=1, activation='sigmoid', name='output')(avg_hz_lstm)

    if opts.use_char:
        model = Model(input=[word_input, char_input], output=y)
    else:
        model = Model(input=word_input, output=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    start_time = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f s" % total_time)

    return model


def build_char_stacked_model(opts, char_vocabsize=0, maxnum=50, maxlen=50, maxcharlen=20, verbose=False, init_mean_value=None):
    # LSTM stacked over CNN based on sentence level
    # features are purely char-based CNN features
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, nbfilters = %s, filter1_len = %s, drop rate = %s" % (N, L,
        opts.nbfilters, opts.filter1_len, opts.dropout))

    # word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    # x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
    # x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    # drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    # resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    # add char-based CNN, concatenating with word embedding to compose word representation
    char_input = Input(shape=(N*L*maxcharlen,), dtype='int32', name='char_input')
    xc = Embedding(output_dim=opts.char_embedd_dim, input_dim=char_vocabsize, input_length=N*L*maxcharlen, mask_zero=True, name='xc')(char_input)
    xc_masked = ZeroMaskedEntries(name='xc_masked')(xc)
    drop_xc = Dropout(opts.dropout, name='drop_xc')(xc_masked)
    res_xc = Reshape((N*L, maxcharlen, opts.char_embedd_dim), name='res_xc')(drop_xc)
    cnn_xc = TimeDistributed(Convolution1D(opts.char_nbfilters, opts.filter2_len, border_mode='valid'), name='cnn_xc')(res_xc)
    max_xc = TimeDistributed(GlobalMaxPooling1D(), name='max_xc')(cnn_xc)
    avg_xc = TimeDistributed(GlobalAveragePooling1D(), name='avg_xc')(cnn_xc)
    res_avg_xc = Reshape((N, L, opts.char_nbfilters), name='res_avg_xc')(avg_xc)
    res_max_xc = Reshape((N, L, opts.char_nbfilters), name='res_max_xc')(max_xc)
    w_repr = merge([res_avg_xc, res_max_xc], mode='concat', name='w_repr')
    zcnn = TimeDistributed(Convolution1D(opts.nbfilters, opts.filter1_len, border_mode='valid'), name='zcnn')(w_repr)

    # pooling mode
    if opts.mode == 'mot':
        logger.info("Use mean-over-time pooling on sentence")
        avg_zcnn = TimeDistributed(GlobalAveragePooling1D(), name='avg_zcnn')(zcnn)
    elif opts.mode == 'att':
        logger.info('Use attention-pooling on sentence')
        avg_zcnn = TimeDistributed(Attention(), name='avg_zcnn')(zcnn)
    elif opts.mode == 'merged':
        logger.info('Use mean-over-time and attention-pooling together on sentence')
        avg_zcnn1 = TimeDistributed(GlobalAveragePooling1D(), name='avg_zcnn1')(zcnn)
        avg_zcnn2 = TimeDistributed(Attention(), name='avg_zcnn2')(zcnn)
        avg_zcnn = merge([avg_zcnn1, avg_zcnn2], mode='concat', name='avg_zcnn')
    else:
        raise NotImplementedError
    hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)

    if opts.mode == 'mot':
        logger.info('Use mean-over-time pooling on text')
        avg_hz_lstm = GlobalAveragePooling1D(name='avg_hz_lstm')(hz_lstm)
    elif opts.mode == 'att':
        logger.info('Use attention-pooling on text')
        avg_hz_lstm = Attention(name='avg_hz_lstm')(hz_lstm)
    elif opts.mode == 'merged':
        logger.info('Use mean-over-time and attention-pooling together on text')
        avg_hz_lstm1 = GlobalAveragePooling1D(name='avg_hz_lstm1')(hz_lstm)
        avg_hz_lstm2 = Attention(name='avg_hz_lstm2')(hz_lstm)
        avg_hz_lstm = merge([avg_hz_lstm1, avg_hz_lstm2], mode='concat', name='avg_hz_lstm')
    else:
        raise NotImplementedError
    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(output_dim=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_lstm)
    else:
        y = Dense(output_dim=1, activation='sigmoid', name='output')(avg_hz_lstm)

    if opts.use_char:
        model = Model(input=char_input, output=y)
    # else:
        # model = Model(input=word_input, output=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    start_time = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f s" % total_time)

    return model


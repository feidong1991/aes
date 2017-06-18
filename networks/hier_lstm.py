# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-09 16:19:32
# @Last Modified by:   feidong1991
# @Last Modified time: 2017-06-18 16:06:12
import keras.backend as K
from keras.models import *
from keras.optimizers import *
from keras.layers.core import *
from keras.layers import Input, Embedding, LSTM, GRU, Dense, merge
from keras.layers import TimeDistributed

from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling1D
from utils import get_logger
import time

"""
The following functions contains serveral hierarchical LSTM models for automatic essay scoring:
(1)build_model: hierarchical LSTM (uni-directional) with mean-over-time pooling
(2)build_bidirectional_model: hierarchical BiLSTM
(3)build_attention_model: hierarchical LSTM with mean-over-time pooling on words and attention pooling over sentences
(3)build_attention2_model: hierarchical LSTM with attention pooling over both words and sentences
"""


def build_model(opts, vocab_size=0, maxnum=50, maxlen=50, embedd_dim=50, embedding_weights=None, verbose=False, init_mean_value=None):

    N = maxnum
    L = maxlen
    logger = get_logger("Build model")
    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, lstm_units = %s, drop rate = %s, l2 = %s" % (N, L, embedd_dim,
        opts.lstm_units, opts.dropout, opts.l2_value))
    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, name='x')(word_input)
    drop_x = Dropout(opts.dropout, name='drop_x')(x)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    z = TimeDistributed(LSTM(opts.lstm_units, return_sequences=True), name='z')(resh_W)
    avg_z = TimeDistributed(GlobalAveragePooling1D(), name='avg_z')(z)

    hz = LSTM(opts.lstm_units, return_sequences=True, name='hz')(avg_z)
    # TODO, random drop sentences
    drop_hz = Dropout(opts.dropout, name='drop_hz')(hz)
    avg_hz = GlobalAveragePooling1D(name='avg_hz')(drop_hz)
    y = Dense(output_dim=1, activation='sigmoid', name='output')(avg_hz)

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


def build_bidirectional_model(opts, vocab_size=0, maxnum=50, maxlen=50, embedd_dim=50, embedding_weights=None, verbose=False, init_mean_value=None):

    N = maxnum
    L = maxlen
    logger = get_logger("Build bidirectional model")
    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, lstm_units = %s, drop rate = %s, l2 = %s" % (N, L, embedd_dim,
        opts.lstm_units, opts.dropout, opts.l2_value))
    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, name='x')(word_input)
    drop_x = Dropout(opts.dropout, name='drop_x')(x)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    z_fwd = TimeDistributed(LSTM(opts.lstm_units, return_sequences=True), name='z_fwd')(resh_W)
    z_bwd = TimeDistributed(LSTM(opts.lstm_units, return_sequences=True, go_backwards=True), name='z_bwd')(resh_W)
    z_merged = merge([z_fwd, z_bwd], mode='concat', name='z_merged')

    avg_z = TimeDistributed(GlobalAveragePooling1D(), name='avg_z')(z_merged)

    hz_fwd = LSTM(opts.lstm_units, return_sequences=True, name='hz_fwd')(avg_z)
    hz_bwd = LSTM(opts.lstm_units, return_sequences=True, go_backwards=True, name='hz_bwd')(avg_z)
    hz_merged = merge([hz_fwd, hz_bwd], mode='concat', name='hz_merged')
    # avg_h = MeanOverTime(mask_zero=True, name='avg_h')(hz)
    avg_hz = GlobalAveragePooling1D(name='avg_hz')(hz_merged)
    y = Dense(output_dim=1, activation='sigmoid', name='output')(avg_hz)

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


from softattention import Attention

def build_attention_model(opts, vocab_size=0, maxnum=50, maxlen=50, embedd_dim=50, embedding_weights=None, verbose=False, init_mean_value=None):
    N = maxnum
    L = maxlen

    logger = get_logger('Build attention pooling model')
    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, lstm_units = %s, drop rate = %s, l2 = %s" % (N, L, embedd_dim,
        opts.lstm_units, opts.dropout, opts.l2_value))
    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, name='x')(word_input)
    drop_x = Dropout(opts.dropout, name='drop_x')(x)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    z = TimeDistributed(LSTM(opts.lstm_units, return_sequences=True), name='z')(resh_W)
    avg_z = TimeDistributed(GlobalAveragePooling1D(), name='avg_z')(z)

    hz = LSTM(opts.lstm_units, return_sequences=True, name='hz')(avg_z)
    # avg_h = MeanOverTime(mask_zero=True, name='avg_h')(hz)
    # avg_hz = GlobalAveragePooling1D(name='avg_hz')(hz)
    attent_hz = Attention(name='attent_hz')(hz)
    y = Dense(output_dim=1, activation='sigmoid', name='output')(attent_hz)

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


def build_attention2_model(opts, vocab_size=0, maxnum=50, maxlen=50, embedd_dim=50, embedding_weights=None, verbose=False, init_mean_value=None):
    N = maxnum
    L = maxlen

    logger = get_logger('Build attention pooling model')
    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, lstm_units = %s, drop rate = %s, l2 = %s" % (N, L, embedd_dim,
        opts.lstm_units, opts.dropout, opts.l2_value))
    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, name='x')(word_input)
    drop_x = Dropout(opts.dropout, name='drop_x')(x)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    z = TimeDistributed(LSTM(opts.lstm_units, return_sequences=True), name='z')(resh_W)
    att_z = TimeDistributed(Attention(name='att_z'))(z)

    hz = LSTM(opts.lstm_units, return_sequences=True, name='hz')(att_z)
    # avg_h = MeanOverTime(mask_zero=True, name='avg_h')(hz)
    # avg_hz = GlobalAveragePooling1D(name='avg_hz')(hz)
    attent_hz = Attention(name='attent_hz')(hz)
    y = Dense(output_dim=1, activation='sigmoid', name='output')(attent_hz)

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
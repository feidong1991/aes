# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-09 16:54:09
# @Last Modified by:   feidong1991
# @Last Modified time: 2017-06-18 17:25:06

import os
import sys
import argparse
import random
import numpy as np
from utils import *
from networks.hier_lstm import build_model

from keras.callbacks import ModelCheckpoint
from metrics import pearson
from metrics import spearman
from metrics import kappa
from metrics import mean_square_error, root_mean_square_error
# from keras.utils.visualize_util import plot
from utils import rescale_tointscore
from utils import domain_specific_rescale
from data_prepare import prepare_sentence_data
from evaluator import Evaluator
import time

logger = get_logger("Train sentence sequences sents-HiLSTM")
np.random.seed(100)


def main():
    parser = argparse.ArgumentParser(description="sentence Hi_LSTM-attent-pooling model")
    parser.add_argument('--train_flag', action='store_true', help='Train or eval')
    parser.add_argument('--fine_tune', action='store_true', help='Fine tune word embeddings')
    parser.add_argument('--embedding', type=str, default='word2vec', help='Word embedding type, word2vec, senna or glove')
    parser.add_argument('--embedding_dict', type=str, default=None, help='Pretrained embedding path')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Only useful when embedding is randomly initialised')

    parser.add_argument('--use_char', action='store_true', help='Whether use char embedding or not')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of texts in each batch')
    parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")

    parser.add_argument('--rnn_type', type=str, default='LSTM', help='Recurrent type')
    parser.add_argument('--lstm_units', type=int, default=100, help='Num of hidden units in recurrent layer')

    # parser.add_argument('--project_hiddensize', type=int, default=100, help='num of units in projection layer')
    parser.add_argument('--optimizer', choices=['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop'], help='updating algorithm', default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for layers')
    parser.add_argument('--oov', choices=['random', 'embedding'], help="Embedding for oov word", required=True)
    parser.add_argument('--l2_value', type=float, default=0.001, help='l2 regularizer value')
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint directory')

    parser.add_argument('--train')  # "data/word-level/*.train"
    parser.add_argument('--dev')
    parser.add_argument('--test')
    parser.add_argument('--prompt_id', type=int, default=1, help='prompt id of essay set')
    parser.add_argument('--bidirectional', action='store_true', help='bidirectional RNN or not')
    parser.add_argument('--masked', action='store_true', help='using mask')

    # to support init bias of output layer of the network
    parser.add_argument('--init_bias', action='store_true', help='initial bias of output' + 
                        'layer with the mean score of training data')

    args = parser.parse_args()

    train_flag = args.train_flag
    fine_tune = args.fine_tune
    USE_CHAR = args.use_char

    batch_size = args.batch_size
    checkpoint_dir = args.checkpoint_path
    num_epochs = args.num_epochs

    bidirectional = args.bidirectional
    if bidirectional:
        modelname = "sent_hibilstm-attent.prompt%s.%sunits.bs%s.hdf5" % (args.prompt_id, args.lstm_units, batch_size)
        imgname = "sent_hibilstm-attent.prompt%s.%sunits.bs%s.png" % (args.prompt_id, args.lstm_units, batch_size)
    else:
        modelname = "sent_hilstm-attent.prompt%s.%sunits.bs%s.hdf5" % (args.prompt_id, args.lstm_units, batch_size)
        imgname = "sent_hilstm-attent.prompt%s.%sunits.bs%s.png" % (args.prompt_id, args.lstm_units, batch_size)

    if args.masked:
        modelname = 'masked_' + modelname
        imgname = 'masked_' + imgname

    if USE_CHAR:
        modelname = 'char_' + modelname
        imgname = 'char_' + imgname
    modelpath = os.path.join(checkpoint_dir, modelname)
    imgpath = os.path.join(checkpoint_dir, imgname)

    datapaths = [args.train, args.dev, args.test]
    embedding_path = args.embedding_dict
    oov = args.oov
    embedding = args.embedding
    embedd_dim = args.embedding_dim
    prompt_id = args.prompt_id

    (X_train, Y_train, mask_train), (X_dev, Y_dev, mask_dev), (X_test, Y_test, mask_test), \
            vocab, vocab_size, embed_table, overal_maxlen, overal_maxnum, init_mean_value = prepare_sentence_data(datapaths, \
            embedding_path, embedding, embedd_dim, prompt_id, args.vocab_size, tokenize_text=True, \
            to_lower=True, sort_by_len=False, vocab_path=None, score_index=6)

    # print type(embed_table)
    if embed_table is not None:
        embedd_dim = embed_table.shape[1]
        embed_table = [embed_table]
        
    max_sentnum = overal_maxnum
    max_sentlen = overal_maxlen
    # print embed_table
    # print X_train[0, 0:10, :]
    # print Y_train[0:10]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
    X_dev = X_dev.reshape((X_dev.shape[0], X_dev.shape[1]*X_dev.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

    logger.info("X_train shape: %s" % str(X_train.shape))
    if USE_CHAR:
        raise NotImplementedError

    else:
        if args.masked:
            model = masked_hilstm.build_model(args, vocab_size, max_sentnum, max_sentlen, embedd_dim, embed_table, True, init_mean_value)
        else:
            model = build_model(args, vocab_size, max_sentnum, max_sentlen, embedd_dim, embed_table, True, init_mean_value)

    # TODO here, modified to evaluate every epoch
    evl = Evaluator(args.prompt_id, False, checkpoint_dir, modelname, X_train, X_dev, X_test, None, None, None, Y_train, Y_dev, Y_test, False)
    
    # Initial evaluation
    logger.info("Initial evaluation: ")
    evl.evaluate(model, -1, print_info=True)
    logger.info("Train model")
    for ii in xrange(args.num_epochs):
        logger.info('Epoch %s/%s' % (str(ii+1), args.num_epochs))
        start_time = time.time()
        model.fit(X_train, Y_train, batch_size=args.batch_size, nb_epoch=1, verbose=0, shuffle=True)
        tt_time = time.time() - start_time
        logger.info("Training one epoch in %.3f s" % tt_time)
        evl.evaluate(model, ii+1)
        evl.print_info()

    evl.print_final_info()
    # use nea evaluator, verified to be identical as above results
    # from asap_evaluator import Evaluator
    # out_dir = args.checkpoint_path
    # evl = Evaluator(reader, args.prompt_id, out_dir, X_dev, X_test, y_dev_pred, y_test_pred, Y_dev_org, Y_test_org)
    # dev_qwk, test_qwk, dev_lwk, test_lwk = evl.calc_qwk(y_dev_pred, y_test_pred)
    # dev_prs, test_prs, dev_spr, test_spr, dev_tau, test_tau = evl.calc_correl(y_dev_pred, y_test_pred)

    # logger.info("Use nea evaluator")
    # logger.info("dev pearson = %s, test pearson = %s" %(dev_prs, test_prs))
    # logger.info("dev spearman = %s, test spearman = %s" %(dev_spr, test_spr))
    # logger.info('dev kappa = %s, test kappa = %s' % (dev_qwk, test_qwk))


if __name__ == '__main__':
    main()


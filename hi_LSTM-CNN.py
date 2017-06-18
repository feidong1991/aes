# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-10 11:57:22
# @Last Modified by:   feidong1991
# @Last Modified time: 2017-06-18 16:44:56


import os
import sys
import argparse
import random
import time
import numpy as np
from utils import *
from networks.hier_networks import build_hrcnn_model
from keras.callbacks import ModelCheckpoint
from metrics import pearson
from metrics import spearman
from metrics import kappa
from metrics import mean_square_error, root_mean_square_error
# from keras.utils.visualize_util import plot
from utils import rescale_tointscore
from utils import domain_specific_rescale
import data_prepare
from evaluator import Evaluator

logger = get_logger("Train sentence sequences Recurrent Convolutional model (LSTM stack over CNN)")
np.random.seed(100)


def main():
    parser = argparse.ArgumentParser(description="sentence Hi_CNN model")
    parser.add_argument('--train_flag', action='store_true', help='Train or eval')
    parser.add_argument('--fine_tune', action='store_true', help='Fine tune word embeddings')
    parser.add_argument('--embedding', type=str, default='word2vec', help='Word embedding type, word2vec, senna or glove')
    parser.add_argument('--embedding_dict', type=str, default=None, help='Pretrained embedding path')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Only useful when embedding is randomly initialised')
    parser.add_argument('--char_embedd_dim', type=int, default=30, help='char embedding dimension if using char embedding')

    parser.add_argument('--use_char', action='store_true', help='Whether use char embedding or not')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of texts in each batch')
    parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")

    parser.add_argument('--nbfilters', type=int, default=100, help='Num of filters in conv layer')
    parser.add_argument('--char_nbfilters', type=int, default=20, help='Num of char filters in conv layer')
    parser.add_argument('--filter1_len', type=int, default=5, help='filter length in 1st conv layer')
    parser.add_argument('--filter2_len', type=int, default=3, help='filter length in 2nd conv layer or char conv layer')
    parser.add_argument('--rnn_type', type=str, default='LSTM', help='Recurrent type')
    parser.add_argument('--lstm_units', type=int, default=100, help='Num of hidden units in recurrent layer')

    # parser.add_argument('--project_hiddensize', type=int, default=100, help='num of units in projection layer')
    parser.add_argument('--optimizer', choices=['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop'], help='updating algorithm', default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for layers')
    parser.add_argument('--oov', choices=['random', 'embedding'], help="Embedding for oov word", required=True)
    parser.add_argument('--l2_value', type=float, help='l2 regularizer value')
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint directory')

    parser.add_argument('--train')  # "data/word-level/*.train"
    parser.add_argument('--dev')
    parser.add_argument('--test')
    parser.add_argument('--prompt_id', type=int, default=1, help='prompt id of essay set')
    parser.add_argument('--init_bias', action='store_true', help='init the last layer bias with average score of training data')
    parser.add_argument('--mode', type=str, choices=['mot', 'att', 'merged'], default='mot', \
                        help='Mean-over-Time pooling or attention-pooling, or two pooling merged')

    args = parser.parse_args()

    train_flag = args.train_flag
    fine_tune = args.fine_tune
    USE_CHAR = args.use_char

    batch_size = args.batch_size
    checkpoint_dir = args.checkpoint_path
    num_epochs = args.num_epochs

    modelname = "sent_hiclstm-%s.prompt%s.%sfilters.bs%s.hdf5" % (args.mode, args.prompt_id, args.nbfilters, batch_size)
    imgname = "sent_hiclstm-%s.prompt%s.%sfilters.bs%s.png" % (args.mode, args.prompt_id, args.nbfilters, batch_size)

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

    # debug mode
    # debug = True
    # if debug:
    # 	nn_model = build_concat_model(args, args.vocab_size, 71, 20, embedd_dim, None, True)

    if args.use_char:
        (X_train, C_train, Y_train, mask_train), (X_dev, C_dev, Y_dev, mask_dev), (X_test, C_test, Y_test, mask_test), \
                vocab, vocab_size, char_vocab, char_vocabsize, embed_table, overal_maxlen, overal_maxnum, maxcharlen, init_mean_value = data_prepare.prepare_data(datapaths, \
                embedding_path, embedding, embedd_dim, prompt_id, args.vocab_size, tokenize_text=True, \
                to_lower=True, sort_by_len=False, vocab_path=None, score_index=6)
    else:
        (X_train, Y_train, mask_train), (X_dev, Y_dev, mask_dev), (X_test, Y_test, mask_test), \
                vocab, vocab_size, embed_table, overal_maxlen, overal_maxnum, init_mean_value = data_prepare.prepare_sentence_data(datapaths, \
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
    # print C_train[0, 0, 0, :], C_train[0, 0, 1, :], C_train[0, 0, -1, :]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
    X_dev = X_dev.reshape((X_dev.shape[0], X_dev.shape[1]*X_dev.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
    logger.info("X_train shape: %s" % str(X_train.shape))

    if not args.use_char:
        C_train, C_dev, C_test = None, None, None
        char_vocabsize = 0
        maxcharlen = 0
    else:
        C_train = C_train.reshape((C_train.shape[0], C_train.shape[1]*C_train.shape[2]*C_train.shape[3]))
        C_dev = C_dev.reshape((C_dev.shape[0], C_dev.shape[1]*C_dev.shape[2]*C_dev.shape[3]))
        C_test = C_test.reshape((C_test.shape[0], C_test.shape[1]*C_test.shape[2]*C_test.shape[3]))

        logger.info("C_train shape: %s" % str(C_train.shape))

    model = build_hrcnn_model(args, vocab_size, char_vocabsize+1, max_sentnum, max_sentlen, maxcharlen, embedd_dim, embed_table, True, init_mean_value)

    evl = Evaluator(args.prompt_id, args.use_char, checkpoint_dir, modelname, X_train, X_dev, X_test, C_train, C_dev, C_test, Y_train, Y_dev, Y_test)

    # Initial evaluation
    logger.info("Initial evaluation: ")
    evl.evaluate(model, -1, print_info=True)
    logger.info("Train model")
    for ii in xrange(args.num_epochs):
        logger.info('Epoch %s/%s' % (str(ii+1), args.num_epochs))
        start_time = time.time()
        if args.use_char:
            model.fit([X_train, C_train], Y_train, batch_size=args.batch_size, nb_epoch=1, verbose=0, shuffle=True)
        else:
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


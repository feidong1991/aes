# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-05 20:15:33
# @Last Modified by:   feidong1991
# @Last Modified time: 2017-02-19 15:59:38

import random
import codecs
import sys
import nltk
# import logging
import re
import numpy as np
import pickle as pk
import utils

url_replacer = '<url>'
logger = utils.get_logger("Loading data...")
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'

MAX_SENTLEN = 50
MAX_SENTNUM = 100

asap_ranges = {
    0: (0, 60),
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60)
}


def get_ref_dtype():
    return ref_scores_dtype


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens


def get_score_range(prompt_id):
    return asap_ranges[prompt_id]


def get_model_friendly_scores(scores_array, prompt_id_array):
    arg_type = type(prompt_id_array)
    assert arg_type in {int, np.ndarray}
    if arg_type is int:
        low, high = asap_ranges[prompt_id_array]
        scores_array = (scores_array - low) / (high - low)
    # else:
    #     assert scores_array.shape[0] == prompt_id_array.shape[0]
    #     dim = scores_array.shape[0]
    #     low = np.zeros(dim)
    #     high = np.zeros(dim)
    #     for ii in range(dim):
    #         low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
    #     scores_array = (scores_array - low) / (high - low)
    # assert np.all(scores_array >= 0) and np.all(scores_array <= 1)
    return scores_array


def convert_to_dataset_friendly_scores(scores_array, prompt_id_array):
    arg_type = type(prompt_id_array)
    assert arg_type in {int, np.ndarray}
    if arg_type is int:
        low, high = asap_ranges[prompt_id_array]
        scores_array = scores_array * (high - low) + low
        assert np.all(scores_array >= low) and np.all(scores_array <= high)
    else:
        assert scores_array.shape[0] == prompt_id_array.shape[0]
        dim = scores_array.shape[0]
        low = np.zeros(dim)
        high = np.zeros(dim)
        for ii in range(dim):
            low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
        scores_array = scores_array * (high - low) + low
    return scores_array


def is_number(token):
    return bool(num_regex.match(token))


def load_vocab(vocab_path):
    logger.info('Loading vocabulary from: ' + vocab_path)
    with open(vocab_path, 'rb') as vocab_file:
        vocab = pk.load(vocab_file)
    return vocab


def create_vocab(file_path, prompt_id, vocab_size, tokenize_text, to_lower):
    logger.info('Creating vocabulary from: ' + file_path)
    total_words, unique_words = 0, 0
    word_freqs = {}
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[6])
            if essay_set == prompt_id or prompt_id <= 0:
                if tokenize_text:
                    content = text_tokenizer(content, True, True, True)
                if to_lower:
                    content = [w.lower() for w in content]
                for word in content:
                    try:
                        word_freqs[word] += 1
                    except KeyError:
                        unique_words += 1
                        word_freqs[word] = 1
                    total_words += 1
    logger.info('  %i total words, %i unique words' % (total_words, unique_words))
    import operator
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
    if vocab_size <= 0:
        # Choose vocab size automatically by removing all singletons
        vocab_size = 0
        for word, freq in sorted_word_freqs:
            if freq > 1:
                vocab_size += 1
    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    vcb_len = len(vocab)
    index = vcb_len
    for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
        vocab[word] = index
        index += 1
    return vocab


def create_char_vocab(file_path, prompt_id, tokenize_text, to_lower):
    logger.info("Create char vocabulary from: %s" % file_path)
    total_chars, unique_chars = 0, 0
    char_vocab = {}
    start_index = 1
    char_vocab['<unk>'] = start_index
    next_index = start_index + 1
    with codecs.open(file_path, 'r', encoding='utf-8') as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[6])
            if essay_set == prompt_id or prompt_id <= 0:
                if tokenize_text:
                    content = text_tokenizer(content, True, True, True)
                if to_lower:
                    content = [w.lower() for w in content]
                for word in content:
                    for char in list(word):
                        if not char in char_vocab:
                            char_vocab[char] = next_index
                            next_index += 1
                            unique_chars += 1
                        total_chars += 1
    logger.info('  %i total chars, %i unique chars' % (total_chars, unique_chars))
    return char_vocab


def read_essays(file_path, prompt_id):
    logger.info('Reading tsv from: ' + file_path)
    essays_list = []
    essays_ids = []
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            if int(tokens[1]) == prompt_id or prompt_id <= 0:
                essays_list.append(tokens[2].strip())
                essays_ids.append(int(tokens[0]))
    return essays_list, essays_ids


def replace_url(text):
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
    return replaced_text


def text_tokenizer(text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=False):
    text = replace_url(text)
    text = text.replace(u'"', u'')
    if "..." in text:
        text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)
        # print text
    if "??" in text:
        text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)
        # print text
    if "!!" in text:
        text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)
        # print text

    # TODO here
    tokens = tokenize(text)
    if tokenize_sent_flag:
        text = " ".join(tokens)
        sent_tokens = tokenize_to_sentences(text, MAX_SENTLEN, create_vocab_flag)
        # print sent_tokens
        # sys.exit(0)
        # if not create_vocab_flag:
        #     print "After processed and tokenized, sentence num = %s " % len(sent_tokens)
        return sent_tokens
    else:
        raise NotImplementedError


def tokenize_to_sentences(text, max_sentlength, create_vocab_flag=False):

    # tokenize a long text to a list of sentences
    sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)

    # Note
    # add special preprocessing for abnormal sentence splitting
    # for example, sentence1 entangled with sentence2 because of period "." connect the end of sentence1 and the begin of sentence2
    # see example: "He is running.He likes the sky". This will be treated as one sentence, needs to be specially processed.
    processed_sents = []
    for sent in sents:
        if re.search(r'(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent):
            s = re.split(r'(?=.{2,})(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent)
            # print sent
            # print s
            ss = " ".join(s)
            ssL = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', ss)

            processed_sents.extend(ssL)
        else:
            processed_sents.append(sent)

    if create_vocab_flag:
        sent_tokens = [tokenize(sent) for sent in processed_sents]
        tokens = [w for sent in sent_tokens for w in sent]
        # print tokens
        return tokens

    # TODO here
    sent_tokens = []
    for sent in processed_sents:
        shorten_sents_tokens = shorten_sentence(sent, max_sentlength)
        sent_tokens.extend(shorten_sents_tokens)
    # if len(sent_tokens) > 90:
    #     print len(sent_tokens), sent_tokens
    return sent_tokens


def shorten_sentence(sent, max_sentlen):
    # handling extra long sentence, truncate to no more extra max_sentlen
    new_tokens = []
    sent = sent.strip()
    tokens = nltk.word_tokenize(sent)
    if len(tokens) > max_sentlen:
        # print len(tokens)
        # Step 1: split sentence based on keywords
        # split_keywords = ['because', 'but', 'so', 'then', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        split_keywords = ['because', 'but', 'so', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
        processed_tokens = []
        if not k_indexes:
            num = len(tokens) / max_sentlen
            k_indexes = [(i+1)*max_sentlen for i in xrange(num)]

        processed_tokens.append(tokens[0:k_indexes[0]])
        len_k = len(k_indexes)
        for j in xrange(len_k-1):
            processed_tokens.append(tokens[k_indexes[j]:k_indexes[j+1]])
        processed_tokens.append(tokens[k_indexes[-1]:])

        # Step 2: split sentence to no more than max_sentlen
        # if there are still sentences whose length exceeds max_sentlen
        for token in processed_tokens:
            if len(token) > max_sentlen:
                num = len(token) / max_sentlen
                s_indexes = [(i+1)*max_sentlen for i in xrange(num)]

                len_s = len(s_indexes)
                new_tokens.append(token[0:s_indexes[0]])
                for j in xrange(len_s-1):
                    new_tokens.append(token[s_indexes[j]:s_indexes[j+1]])
                new_tokens.append(token[s_indexes[-1]:])

            else:
                new_tokens.append(token)
    else:
            return [tokens]

    # print "Before processed sentences length = %d, after processed sentences num = %d " % (len(tokens), len(new_tokens))
    return new_tokens


def read_dataset(file_path, prompt_id, vocab, to_lower, score_index=6, char_level=False):
    logger.info('Reading dataset from: ' + file_path)

    data_x, data_y, prompt_ids = [], [], []
    num_hit, unk_hit, total = 0., 0., 0.
    max_sentnum = -1
    max_sentlen = -1
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[score_index])
            if essay_set == prompt_id or prompt_id <= 0:
                # tokenize text into sentences
                sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
                if to_lower:
                    sent_tokens = [[w.lower() for w in s] for s in sent_tokens]
                if char_level:
                    raise NotImplementedError

                sent_indices = []
                indices = []
                if char_level:
                    raise NotImplementedError
                else:
                    for sent in sent_tokens:
                        length = len(sent)
                        if max_sentlen < length:
                            max_sentlen = length

                        for word in sent:
                            if is_number(word):
                                indices.append(vocab['<num>'])
                                num_hit += 1
                            elif word in vocab:
                                indices.append(vocab[word])
                            else:
                                indices.append(vocab['<unk>'])
                                unk_hit += 1
                            total += 1
                        sent_indices.append(indices)
                        indices = []
                data_x.append(sent_indices)
                data_y.append(score)
                prompt_ids.append(essay_set)

                if max_sentnum < len(sent_indices):
                    max_sentnum = len(sent_indices)
    logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
    return data_x, data_y, prompt_ids, max_sentlen, max_sentnum


def get_data(paths, prompt_id, vocab_size, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None, score_index=6):
    train_path, dev_path, test_path = paths[0], paths[1], paths[2]

    logger.info("Prompt id is %s" % prompt_id)
    if not vocab_path:
        vocab = create_vocab(train_path, prompt_id, vocab_size, tokenize_text, to_lower)
        if len(vocab) < vocab_size:
            logger.warning('The vocabulary includes only %i words (less than %i)' % (len(vocab), vocab_size))
        else:
            assert vocab_size == 0 or len(vocab) == vocab_size
    else:
        vocab = load_vocab(vocab_path)
        if len(vocab) != vocab_size:
            logger.warning('The vocabulary includes %i words which is different from given: %i' % (len(vocab), vocab_size))
    logger.info('  Vocab size: %i' % (len(vocab)))

    train_x, train_y, train_prompts, train_maxsentlen, train_maxsentnum = read_dataset(train_path, prompt_id, vocab, to_lower)
    dev_x, dev_y, dev_prompts, dev_maxsentlen, dev_maxsentnum = read_dataset(dev_path, prompt_id, vocab, to_lower)
    test_x, test_y, test_prompts, test_maxsentlen, test_maxsentnum = read_dataset(test_path, prompt_id, vocab,  to_lower)

    overal_maxlen = max(train_maxsentlen, dev_maxsentlen, test_maxsentlen)
    overal_maxnum = max(train_maxsentnum, dev_maxsentnum, test_maxsentnum)

    logger.info("Training data max sentence num = %s, max sentence length = %s" % (train_maxsentnum, train_maxsentlen))
    logger.info("Dev data max sentence num = %s, max sentence length = %s" % (dev_maxsentnum, dev_maxsentlen))
    logger.info("Test data max sentence num = %s, max sentence length = %s" % (test_maxsentnum, test_maxsentlen))
    logger.info("Overall max sentence num = %s, max sentence length = %s" % (overal_maxnum, overal_maxlen))

    return (train_x, train_y, train_prompts), (dev_x, dev_y, dev_prompts), (test_x, test_y, test_prompts), vocab, overal_maxlen, overal_maxnum


def read_char_dataset(file_path, prompt_id, vocab, char_vocab, to_lower, score_index=6, char_level=True):
    # support reading chars
    logger.info('Reading dataset from: ' + file_path)

    data_x, data_y, char_x, prompt_ids = [], [], [], []
    num_hit, unk_hit, total = 0., 0., 0.
    max_sentnum = -1
    max_sentlen = -1
    maxcharlen = -1
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[score_index])
            if essay_set == prompt_id or prompt_id <= 0:
                # tokenize text into sentences
                sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
                if to_lower:
                    sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

                sent_indices = []
                indices = []
                chars_indices = []
                wc_indices = []

                if char_level:
                    # raise NotImplementedError
                # else:
                    for sent in sent_tokens:
                        length = len(sent)
                        if max_sentlen < length:
                            max_sentlen = length

                        for word in sent:
                            if is_number(word):
                                indices.append(vocab['<num>'])
                                num_hit += 1
                            elif word in vocab:
                                indices.append(vocab[word])
                            else:
                                indices.append(vocab['<unk>'])
                                unk_hit += 1
                            total += 1
                            # chars
                            c_indices = []
                            current_chars = list(word)
                            if len(current_chars) > maxcharlen:
                                maxcharlen = len(current_chars)
                            for c in current_chars:
                                try:
                                    c_indices.append(char_vocab[c])
                                except:
                                    c_indices.append(char_vocab['<unk>'])
                            wc_indices.append(c_indices)
                        chars_indices.append(wc_indices)
                        sent_indices.append(indices)
                        indices = []
                        wc_indices = []
                data_x.append(sent_indices)
                data_y.append(score)
                char_x.append(chars_indices)
                prompt_ids.append(essay_set)

                if max_sentnum < len(sent_indices):
                    max_sentnum = len(sent_indices)
    logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
    return data_x, char_x,  data_y, prompt_ids, max_sentlen, max_sentnum, maxcharlen


def get_char_data(paths, prompt_id, vocab_size, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None, score_index=6):
    train_path, dev_path, test_path = paths[0], paths[1], paths[2]

    logger.info("Prompt id is %s" % prompt_id)
    if not vocab_path:
        vocab = create_vocab(train_path, prompt_id, vocab_size, tokenize_text, to_lower)
        char_vocab = create_char_vocab(train_path, prompt_id, tokenize_text, to_lower)
        if len(vocab) < vocab_size:
            logger.warning('The vocabulary includes only %i words (less than %i)' % (len(vocab), vocab_size))
        else:
            assert vocab_size == 0 or len(vocab) == vocab_size
    else:
        vocab = load_vocab(vocab_path)
        if len(vocab) != vocab_size:
            logger.warning('The vocabulary includes %i words which is different from given: %i' % (len(vocab), vocab_size))
    logger.info(' Word vocab size: %i, char vocab size: %i' % (len(vocab), len(char_vocab)))

    train_x, train_char_x, train_y, train_prompts, train_maxsentlen, train_maxsentnum, train_charlen = read_char_dataset(train_path, prompt_id, vocab, char_vocab, to_lower)
    dev_x, dev_char_x, dev_y, dev_prompts, dev_maxsentlen, dev_maxsentnum, dev_charlen = read_char_dataset(dev_path, prompt_id, vocab, char_vocab, to_lower)
    test_x, test_char_x, test_y, test_prompts, test_maxsentlen, test_maxsentnum, test_charlen = read_char_dataset(test_path, prompt_id, vocab, char_vocab, to_lower)

    overal_maxlen = max(train_maxsentlen, dev_maxsentlen, test_maxsentlen)
    overal_maxnum = max(train_maxsentnum, dev_maxsentnum, test_maxsentnum)
    maxcharlen = max(train_charlen, dev_charlen, test_charlen)

    logger.info("Training data max sentence num = %s, max sentence length = %s, max char len = %s" % (train_maxsentnum, train_maxsentlen, train_charlen))
    logger.info("Dev data max sentence num = %s, max sentence length = %s, max char len = %s" % (dev_maxsentnum, dev_maxsentlen, dev_charlen))
    logger.info("Test data max sentence num = %s, max sentence length = %s, max char len = %s" % (test_maxsentnum, test_maxsentlen, test_charlen))
    logger.info("Overall max sentence num = %s, max sentence length = %s, max char len = %s" % (overal_maxnum, overal_maxlen, maxcharlen))

    return (train_x, train_char_x, train_y, train_prompts), (dev_x, dev_char_x, dev_y, dev_prompts), (test_x, test_char_x, test_y, test_prompts), \
            vocab, char_vocab, overal_maxlen, overal_maxnum, maxcharlen
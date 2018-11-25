# -*- coding: utf-8 -*-

import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from config import Config

config = Config()


def pickle_load(file_path):
    return pickle.load(open(file_path, 'rb'))


def pickle_dump(obj, file_path):
    pickle.dump(obj, open(file_path, 'wb'))


def get_score_joint(y_true, y_pred):
    """
    return score for predictions made by joint model
    :param y_true: list of arrays shaped [batch_size, 4], 20 arrays in all
    :param y_pred: list of arrays shaped [batch_size, 4], 20 arrays in all
    :return:
    """
    tp = 0
    fp = 0
    fn = 0

    for i in range(len(y_true)):    # len(y_true) is the number of aspects
        if len(y_true[i].shape) == 2:
            true = np.argmax(y_true[i], axis=-1)
        else:
            true = y_true[i]
        if len(y_pred[i].shape) == 2:
            pred = np.argmax(y_pred[i], axis=-1)
        else:
            pred = y_pred[i]
        batch_size = true.shape[0]
        for j in range(batch_size):
            if true[j] != 0 and pred[j] != 0 and true[j] == pred[j]:
                tp += 1
            elif true[j] != 0 and pred[j] != 0 and true[j] != pred[j]:
                fp += 1
            elif true[j] != 0 and pred[j] == 0:
                fn += 1
            elif true[j] == 0 and pred[j] != 0:
                fp += 1

    if tp + fp == 0:
        precision = 1.0
    else:
        precision = float(tp) / (tp + fp)
    if tp + fn == 0:
        recall = 1.0
    else:
        recall = float(tp) / (tp + fn)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall / (precision + recall)

    print('precision: ', precision)
    print('recall: ', recall)
    print('f1_score: ', f1)
    return precision, recall, f1


def get_score_aspect(y_true, y_pred, threshold=0.5):
    """
    return score for predictions made by aspect classification model
    :param y_true: array shaped [batch_size, 20]
    :param y_pred: array shaped [batch_size, 20]
    :param threshold:
    :return:
    """
    y_pred = np.array([[1 if y_pred[i, j] >= threshold else 0 for j in range(y_pred.shape[1])]
                       for i in range(y_pred.shape[0])])
    tp = 0
    fp = 0
    fn = 0

    for i in range(y_true.shape[0]):
        true = y_true[i]
        pred = y_pred[i]
        for j in range(y_pred.shape[1]):
            if true[j] == 1 and pred[j] == 1:
                tp += 1
            elif true[j] == 1 and pred[j] == 0:
                fn += 1
            elif true[j] == 0 and pred[j] == 1:
                fp += 1

    if tp + fp == 0:
        precision = 1.0
    else:
        precision = float(tp) / (tp + fp)
    if tp + fn == 0:
        recall = 1.0
    else:
        recall = float(tp) / (tp + fn)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall / (precision + recall)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1_score: ', f1)
    return precision, recall, f1


def get_score_senti(y_true, y_pred):
    """
    return score for predictions made by sentimant analysis model
    :param y_true: array shaped [batch_size, 3]
    :param y_pred: array shaped [batch_size, 3]
    :return:
    """

    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    acc = accuracy_score(y_true, y_pred)

    print('acc:', acc)
    return acc


def get_score_ensemble(y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0

    if len(y_true.shape) == 2:
        true = np.argmax(y_true, axis=-1)
    else:
        true = y_true
    if len(y_pred.shape) == 2:
        pred = np.argmax(y_pred, axis=-1)
    else:
        pred = y_pred
    batch_size = true.shape[0]
    for j in range(batch_size):
        if true[j] != 0 and pred[j] != 0 and true[j] == pred[j]:
            tp += 1
        elif true[j] != 0 and pred[j] != 0 and true[j] != pred[j]:
            fp += 1
        elif true[j] != 0 and pred[j] == 0:
            fn += 1
        elif true[j] == 0 and pred[j] != 0:
            fp += 1

    if tp + fp == 0:
        precision = 1.0
    else:
        precision = float(tp) / (tp + fp)
    if tp + fn == 0:
        recall = 1.0
    else:
        recall = float(tp) / (tp + fn)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    print('precision: ', precision)
    print('recall: ', recall)
    print('f1_score: ', f1)
    return precision, recall, f1


def pad_sequences_2d(sequences, max_sents, max_words, dtype='int32', padding='pre', truncating='pre', value=0.):
    num_samples = len(sequences)

    x = (np.ones((num_samples, max_sents, max_words)) * value).astype(dtype)

    for i, doc in enumerate(sequences):
        if not len(doc):
            continue    # empty list was found

        if truncating == 'pre':
            doc = doc[-max_sents:]
        elif truncating == 'post':
            doc = doc[:max_sents]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        sents = (np.ones((max_sents, max_words)) * value).astype(dtype)
        for j, sent in enumerate(doc):
            if not len(sent):
                continue

            if truncating == 'pre':
                trunc = sent[-max_words:]
            elif truncating == 'post':
                trunc = sent[:max_words]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)

            trunc = np.asarray(trunc, dtype=dtype)

            if padding == 'post':
                sents[j, :len(trunc)] = trunc
            elif padding == 'pre':
                sents[j, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)

        if padding == 'post':
            x[i, :sents.shape[0], :] = sents
        elif padding == 'pre':
            x[i, -sents.shape[0]:, :] = sents
        else:
            raise ValueError('Padding type "%s" not understood' % padding)

    return x

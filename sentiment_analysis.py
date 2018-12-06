# -*- encoding:utf-8 -*-

import pickle
import os.path
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from util import data_process
from model.model1 import BiLSTM_Attention
import sys


def select_model(modelname, sourcevocabsize, targetvocabsize, word_W,
                 input_seq_lenth, output_seq_lenth, emd_dim,
                 sourcecharsize, char_W, input_word_length, char_emd_dim, batch_size=128):
    nn_model = None

    if modelname is 'BiLSTM_Attention':
        nn_model = BiLSTM_Attention(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                    word_W=word_W,
                                    input_seq_lenth=input_seq_lenth,
                                    output_seq_lenth=output_seq_lenth,
                                    emd_dim=emd_dim,
                                    sourcecharsize=sourcecharsize,
                                    char_W=char_W,
                                    input_word_length=input_word_length,
                                    char_emd_dim=char_emd_dim,
                                    batch_size=batch_size)
    return nn_model


def train_e2e_model(Modelname, datafile, modelfile, resultdir, npochos=100, batch_size=50, retrain=False):
    # load training data and test data
    train, train_char, train_label,\
    test, test_char, test_label,\
    word_vob, vob_idex_word, word_W, word_k,\
    target_vob, vob_idex_target,\
    char_vob, vob_idex_char, char_W, char_k,\
    max_s, max_c = pickle.load(open(datafile, 'rb'))

    # train model
    nn_model = select_model(Modelname, sourcevocabsize=len(word_vob), targetvocabsize=len(target_vob),
                            word_W=word_W, input_seq_lenth=max_s, output_seq_lenth=max_s, emd_dim=word_k,
                            sourcecharsize=len(char_vob), char_W=char_W, input_word_length=max_c,
                            char_emd_dim=char_k, batch_size=batch_size)

    if retrain:
        nn_model.load_weights("./modfile/model1file" + modelfile)

    nn_model.summary()

    indices = np.arange(len(train))
    np.random.shuffle(indices)
    train_shuf = np.zeros((len(train), max_s)).astype('int32')
    train_char_shuf = np.zeros((len(train_char), max_s, max_c)).astype('int32')
    train_label_shuf = np.zeros((len(train_label), len(target_vob))).astype('int32')
    for idx, s in enumerate(indices):
        train_shuf[idx,] = train[s]
        train_char_shuf[idx,] = train_char[s]
        train_label_shuf[idx,] = train_label[s]

    monitor = 'val_acc'  # val_acc val_loss
    early_stopping = EarlyStopping(monitor=monitor, patience=2)
    csv_logger = CSVLogger("./logs/" + modelfile + ".logs")
    checkpointer = ModelCheckpoint(filepath="./modfile/model1file/" + modelfile + ".best_model.h5", monitor=monitor, verbose=0,
                                   save_best_only=False, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=10, min_lr=0.0001)
    nn_model.fit(x=[np.array(train_shuf), np.array(train_char_shuf)],
                 y=np.array(train_label_shuf),
                 batch_size=batch_size,
                 epochs=npochos,
                 verbose=1,
                 shuffle=True,
                 validation_split=0.2,
                 # validation_data=([np.array(test), np.array(test_char)], [np.array(test_label)]),
                 callbacks=[reduce_lr, checkpointer, csv_logger, early_stopping])

    nn_model.save_weights("./modfile/model1file" + modelfile, overwrite=True)

    return nn_model


def evaluate_model(model_name, model_file, batch_size=50):
    train, train_char, train_label, \
    test, test_char, test_label, \
    word_vob, vob_idex_word, word_W, word_k, \
    target_vob, vob_idex_target, \
    char_vob, vob_idex_char, char_W, char_k, \
    max_s, max_c = pickle.load(open(datafile, 'rb'))
    nn_model = select_model(model_name, sourcevocabsize=len(word_vob), targetvocabsize=len(target_vob),
                            word_W=word_W, input_seq_lenth=max_s, output_seq_lenth=max_s, emd_dim=word_k,
                            sourcecharsize=len(char_vob), char_W=char_W, input_word_length=max_c,
                            char_emd_dim=char_k)
    nn_model.load_weights("./modfile/model1file" + model_file)
    loss, acc = nn_model.evaluate([np.array(test), np.array(test_char)], np.array(test_label), verbose=0,
                                  batch_size=batch_size)
    print('\n test_test score:', loss, acc)
    nn_model.load_weights("./modfile/model1file" + model_file + ".best_model.h5")
    loss, acc = nn_model.evaluate([np.array(test), np.array(test_char)], np.array(test_label), verbose=0,
                                  batch_size=batch_size)
    print('bestModel...\n test_test score:', loss, acc)


if __name__ == "__main__":

    maxlen = 100
    batch_size = 128
    npochos = 100
    modelname = 'BiLSTM_Attention'
    trainfile = "./data/mix_data_train_data.json"
    testfile = "./data/mix_data_test_data.json"
    w2v_file = "./modfile/Word2Vec.mod"
    char2v_file = "./modfile/Char2Vec.mod"
    datafile = "./modfile/model1_data/data.pkl"
    modelfile = modelname + ".pkl"
    resultdir = "./modfile/result/"
    print(modelname)

    retrain = True if sys.argv[1] == 'train' else False
    Test = True

    all_data = True
    if all_data:
        if not os.path.exists(datafile):
            print("Precess data....")
            data_process.get_data(trainfile, testfile, w2v_file, char2v_file, datafile, w2v_k=100, c2v_k=100,
                                  maxlen=maxlen)

        if not os.path.exists("./modfile/" + modelfile):
            print("data has extisted: " + datafile)
            print("Training EE model....")
            train_e2e_model(modelname, datafile, modelfile, resultdir,
                            npochos=npochos, batch_size=batch_size, retrain=False)
        else:
            if retrain:
                print("ReTraining EE model....")
                train_e2e_model(modelname, datafile, modelfile, resultdir,
                                npochos=npochos, batch_size=batch_size, retrain=retrain)
        if Test:
            print("test EE model....")
            evaluate_model(modelname, modelfile, batch_size=batch_size)
    else:
        for i in range(0, 5):
            datafile = "./modfile/model1_data/data" + "_fold_" + str(i) + ".pkl"
            modelfile = modelname + "_fold_" + str(i) + ".pkl"

            if not os.path.exists(datafile):
                print("Precess data " + str(i) + "....")
                data_process.get_part_data(trainfile, testfile, w2v_file, char2v_file, datafile, w2v_k=100, c2v_k=100,
                                           maxlen=maxlen, left=i)

            if not os.path.exists("./modfile/" + modelfile):
                print("data has extisted: " + datafile)
                print("Training EE " + str(i) + " model....")
                train_e2e_model(modelname, datafile, modelfile, resultdir,
                                npochos=npochos, batch_size=batch_size, retrain=False)
            else:
                if retrain:
                    print("ReTraining EE " + str(i) + " model....")
                    train_e2e_model(modelname, datafile, modelfile, resultdir,
                                    npochos=npochos, batch_size=batch_size, retrain=retrain)
            if Test:
                print("test EE " + str(i) + " model....")
                evaluate_model(modelname, modelfile, batch_size=batch_size)

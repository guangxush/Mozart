# -*- encoding:utf-8 -*-

import codecs, json, re, pickle
import numpy as np
import sys
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# sys.path.append("..")
# reload(sys)
# sys.setdefaultencoding('utf-8')
import sentiment_analysis
import time, os
from sklearn.ensemble import GradientBoostingClassifier


def make_idx_word_index(s_sent, max_s, max_c, word_vob, char_vob):
    data_s = []
    if len(s_sent) > max_s:
        i = 0
        while i < max_s:
            if not word_vob.__contains__(s_sent[i]):
                data_s.append(word_vob["**UNK**"])
            else:
                data_s.append(word_vob[s_sent[i]])
            i += 1
    else:
        i = 0
        while i < len(s_sent):
            if not word_vob.__contains__(s_sent[i]):
                data_s.append(word_vob["**UNK**"])
            else:
                data_s.append(word_vob[s_sent[i]])
            i += 1
        num = max_s - len(s_sent)
        for inum in range(0, num):
            data_s.append(0)

    data_w = []
    for ii in range(0, min(max_s, len(s_sent))):
        word = s_sent[ii]
        data_c = []
        for chr in range(0, min(word.__len__(), max_c)):
            if not char_vob.__contains__(word[chr]):
                data_c.append(char_vob["**UNK**"])
                # data_c.append(0)
            else:
                data_c.append(char_vob[word[chr]])

        num = max_c - word.__len__()
        for i in range(0, max(num, 0)):
            data_c.append(0)

        data_w.append(data_c)

    num = max_s - len(s_sent)
    for inum in range(0, num):
        data_tmp = []
        for i in range(0, max_c):
            data_tmp.append(0)
        data_w.append(data_tmp)

    return [data_s], [data_w]


def predict_result(model_name, datafile, model_file, testfile):

    train, train_char, train_label, \
    test, test_char, test_label, \
    word_vob, vob_idex_word, word_W, word_k, \
    target_vob, vob_idex_target, \
    char_vob, vob_idex_char, char_W, char_k, \
    max_s, max_c = pickle.load(open(datafile, 'rb'))

    nn_model = sentiment_analysis.select_model(model_name, sourcevocabsize=len(word_vob), targetvocabsize=len(target_vob),
                                               word_W=word_W, input_seq_lenth=max_s, output_seq_lenth=max_s, emd_dim=word_k,
                                               sourcecharsize=len(char_vob), char_W=char_W, input_word_length=max_c,
                                               char_emd_dim=char_k, batch_size=batch_size)

    nn_model.load_weights("./modfile/" + model_file)
    nn_model.summary()
    ft = codecs.open(testfile, 'r', encoding='utf-8')
    lines = ft.readlines()
    t = str(int(time.time()))
    fw = codecs.open("./result/result_temp/classify_result_" + t + ".txt", 'w', encoding='utf-8')
    for num, line in enumerate(lines):
        print(num)
        item = json.loads(line.rstrip('\n'))
        id = item['id']
        words = item['words']
        test_words, test_char = make_idx_word_index(words, max_s, max_c, word_vob, char_vob)
        predictions = nn_model.predict([np.array(test_words),
                                        np.array(test_char)], verbose=0)
        for si in range(0, len(predictions)):
            sent = predictions[si]
            item_p = np.argmax(sent)
            label = vob_idex_target[item_p]
            fw.write(str(id)+'\t'+str(label)+'\n')
    fw.close()


def generate_result(model_name, datafile, model_file, testfile):
    train, train_char, train_label, \
    test, test_char, test_label, \
    word_vob, vob_idex_word, word_W, word_k, \
    target_vob, vob_idex_target, \
    char_vob, vob_idex_char, char_W, char_k, \
    max_s, max_c = pickle.load(open(datafile, 'rb'))
    test_length = len(test_label)
    if not os.path.exists("./modfile/" + model_file):
        print("./modfile/" + model_file + " file not found")
        # if file not exists, return [0]*30
        return np.array([0] * test_length)
    nn_model = sentiment_analysis.select_model(model_name, sourcevocabsize=len(word_vob), targetvocabsize=len(target_vob),
                                               word_W=word_W, input_seq_lenth=max_s, output_seq_lenth=max_s, emd_dim=word_k,
                                               sourcecharsize=len(char_vob), char_W=char_W, input_word_length=max_c,
                                               char_emd_dim=char_k, batch_size=batch_size)

    nn_model.load_weights("./modfile/" + model_file)
    nn_model.summary()
    ft = codecs.open(testfile, 'r', encoding='utf-8')
    lines = ft.readlines()
    result = []
    for num, line in enumerate(lines):
        item = json.loads(line.rstrip('\n'))
        id = item['id']
        words = item['words']
        test_words, test_char = make_idx_word_index(words, max_s, max_c, word_vob, char_vob)
        predictions = nn_model.predict([np.array(test_words),
                                        np.array(test_char)], verbose=0)
        for si in range(0, len(predictions)):
            sent = predictions[si]
            item_p = np.argmax(sent)
            label = vob_idex_target[item_p]
            result.append(label)
    return np.array(result)


def predict_result_mul(model_name, datafile, model_file, testfile):
    train, train_char, train_label, \
    test, test_char, test_label, \
    word_vob, vob_idex_word, word_W, word_k, \
    target_vob, vob_idex_target, \
    char_vob, vob_idex_char, char_W, char_k, \
    max_s, max_c = pickle.load(open(datafile, 'rb'))

    nn_model = sentiment_analysis.select_model(model_name, sourcevocabsize=len(word_vob), targetvocabsize=len(target_vob),
                                               word_W=word_W, input_seq_lenth=max_s, output_seq_lenth=max_s, emd_dim=word_k,
                                               sourcecharsize=len(char_vob), char_W=char_W, input_word_length=max_c,
                                               char_emd_dim=char_k, batch_size=batch_size)
    acc_best = 0.
    acc = 0.
    if os.path.exists("./modfile/" + model_file):
        nn_model.load_weights("./modfile/" + model_file)

        loss, acc = nn_model.evaluate([np.array(test), np.array(test_char)], np.array(test_label), verbose=0,
                                      batch_size=batch_size)
        print('\n test_test score:', loss, acc)

    if os.path.exists("./modfile/" + model_file + ".best_model.h5"):
        nn_model.load_weights("./modfile/" + model_file + ".best_model.h5")
        nn_model.load_weights("./modfile/" + model_file + ".best_model.h5")
        loss, acc_best = nn_model.evaluate([np.array(test), np.array(test_char)], np.array(test_label), verbose=0,
                                           batch_size=batch_size)
        print('bestModel...\n test_test score:', loss, acc_best)

    if acc >= acc_best:
        nn_model.load_weights("./modfile/" + model_file)

    else:
        nn_model.load_weights("./modfile/" + model_file + ".best_model.h5")
        nn_model.summary()

    ft = codecs.open(testfile, 'r', encoding='utf-8')
    lines = ft.readlines()
    t = str(int(time.time()))
    fw = codecs.open("./submission/classify_result_" + t, 'w', encoding='utf-8')
    for num, line in enumerate(lines):
        print(num)
        item = json.loads(line.rstrip('\n'))
        id = item['id']
        words = item['words']
        test_words, test_char = make_idx_word_index(words, max_s, max_c, word_vob, char_vob)
        predictions = nn_model.predict([np.array(test_words),
                                        np.array(test_char)], verbose=0)
        for si in range(0, len(predictions)):
            sent = predictions[si]
            item_p = np.argmax(sent)
            label = vob_idex_target[item_p]
            fw.write(str(id) + '\t' + str(label) + '\n')
    fw.close()


def model_file_select(nn_model, modelfile, test, test_char, test_label):
    target12_acc_best = 0.
    target12_acc = 0.
    if os.path.exists("./modfile/" + modelfile):
        nn_model.load_weights("./modfile/" + modelfile)

        loss, target12_loss, target1_loss, target12_acc, target1_acc = nn_model.evaluate(
            [np.array(test), np.array(test_char)],
            [np.array(test_label)],
            verbose=0, batch_size=batch_size)
        print("\n" + "test score: loss:%.6f, target12_loss:%.6f, target1_loss:%.6f, target12_acc:%.6f, target1_acc:%.6f"
              % (loss, target12_loss, target1_loss, target12_acc, target1_acc))

    if os.path.exists("./modfile/" + modelfile + ".best_model.h5"):
        nn_model.load_weights("./modfile/" + modelfile + ".best_model.h5")
        loss, target12_loss, target1_loss, target12_acc_best, target1_acc = nn_model.evaluate(
            [np.array(test), np.array(test_char)],
            [np.array(test_label)],
            verbose=0, batch_size=batch_size)
        print("\n" + "bestModel test score::loss:%.6f, target12_loss:%.6f, target1_loss:%.6f, target12_acc:%.6f, "
                     "target1_acc:%.6f " % (loss, target12_loss, target1_loss, target12_acc_best, target1_acc))
    if target12_acc >= target12_acc_best:
        nn_model.load_weights("./modfile/" + modelfile)
    else:
        nn_model.load_weights("./modfile/" + modelfile + ".best_model.h5")
        nn_model.summary()
    return nn_model


def predict_submit_task1_merge(modelname1, modelname2, modelname3, modelfile1, modelfile2, modelfile3, datafile,
                               testfile):
    train, train_char, train_label, \
    test, test_char, test_label, \
    word_vob, vob_idex_word, word_W, word_k, \
    target_vob, vob_idex_target, \
    char_vob, vob_idex_char, char_W, char_k, \
    max_s, max_c = pickle.load(open(datafile, 'rb'))

    nn_model1 = sentiment_analysis.select_model(modelname1, sourcevocabsize=len(word_vob), targetvocabsize=len(target_vob),
                                                word_W=word_W, input_seq_lenth=max_s, output_seq_lenth=max_s, emd_dim=word_k,
                                                sourcecharsize=len(char_vob), char_W=char_W, input_word_length=max_c,
                                                char_emd_dim=char_k, batch_size=batch_size)
    model1 = model_file_select(nn_model1, modelfile1, test, test_char, test_label)

    nn_model2 = sentiment_analysis.select_model(modelname2, sourcevocabsize=len(word_vob), targetvocabsize=len(target_vob),
                                                word_W=word_W, input_seq_lenth=max_s, output_seq_lenth=max_s, emd_dim=word_k,
                                                sourcecharsize=len(char_vob), char_W=char_W, input_word_length=max_c,
                                                char_emd_dim=char_k, batch_size=batch_size)

    model2 = model_file_select(nn_model2, modelfile2, test, test_char, test_label)

    nn_model3 = sentiment_analysis.select_model(modelname3, sourcevocabsize=len(word_vob), targetvocabsize=len(target_vob),
                                                word_W=word_W, input_seq_lenth=max_s, output_seq_lenth=max_s, emd_dim=word_k,
                                                sourcecharsize=len(char_vob), char_W=char_W, input_word_length=max_c,
                                                char_emd_dim=char_k, batch_size=batch_size)

    model3 = model_file_select(nn_model3, modelfile3, test, test_char, test_label)

    ft = codecs.open(testfile, 'r', encoding='utf-8')
    lines = ft.readlines()
    t = str(int(time.time()))
    fw = codecs.open("./submission/classify_result_" + t + "_" + "three_task", 'w',
                     encoding='utf-8')

    for num, line in enumerate(lines):
        print(num)
        item = json.loads(line.rstrip('\n'))
        id = item['id']
        words = item['words']
        test_words, test_char = make_idx_word_index(words, max_s, max_c, word_vob, char_vob)
        predictions1 = model1.predict([np.array(test_words),
                                       np.array(test_char)], verbose=0)
        predictions2 = model2.predict([np.array(test_words),
                                       np.array(test_char)], verbose=0)
        predictions3 = model3.predict([np.array(test_words),
                                       np.array(test_char)], verbose=0)
        for si in range(0, len(predictions1[0])):
            item_p = np.argmax(0.33 * predictions1[0][si] + 0.33 * predictions2[0][si] + 0.33 * predictions3[0][si])
            label = vob_idex_target[item_p]
            fw.write(str(id) + '\t' + str(label) + '\n')
    fw.close()


def predict_submit_task1_staking(modelname1, modelname2, modelname3, modelfile1, modelfile2, modelfile3, datafile,
                                 testfile):
    train, train_char, train_label, \
    test, test_char, test_label, \
    word_vob, vob_idex_word, word_W, word_k, \
    target_vob, vob_idex_target, \
    char_vob, vob_idex_char, char_W, char_k, \
    max_s, max_c = pickle.load(open(datafile, 'rb'))

    nn_model1 = sentiment_analysis.select_model(modelname1, sourcevocabsize=len(word_vob), targetvocabsize=len(target_vob),
                                                word_W=word_W, input_seq_lenth=max_s, output_seq_lenth=max_s, emd_dim=word_k,
                                                sourcecharsize=len(char_vob), char_W=char_W, input_word_length=max_c,
                                                char_emd_dim=char_k, batch_size=batch_size)
    model1 = model_file_select(nn_model1, modelfile1, test, test_char, test_label)

    nn_model2 = sentiment_analysis.select_model(modelname2, sourcevocabsize=len(word_vob), targetvocabsize=len(target_vob),
                                                word_W=word_W, input_seq_lenth=max_s, output_seq_lenth=max_s, emd_dim=word_k,
                                                sourcecharsize=len(char_vob), char_W=char_W, input_word_length=max_c,
                                                char_emd_dim=char_k, batch_size=batch_size)

    model2 = model_file_select(nn_model2, modelfile2, test, test_char, test_label)

    nn_model3 = sentiment_analysis.select_model(modelname3, sourcevocabsize=len(word_vob), targetvocabsize=len(target_vob),
                                                word_W=word_W, input_seq_lenth=max_s, output_seq_lenth=max_s, emd_dim=word_k,
                                                sourcecharsize=len(char_vob), char_W=char_W, input_word_length=max_c,
                                                char_emd_dim=char_k, batch_size=batch_size)

    model3 = model_file_select(nn_model3, modelfile3, test, test_char, test_label)
    '''模型融合中使用到的各个单模型'''
    clfs = [model1, model2, model3]

    '''切分一部分数据作为测试集'''
    X, X_predict, y, y_predict = train_test_split(train, train_label, test_size=0.33, random_state=500)
    dataset_blend_train = np.zeros((len(X), len(clfs)))
    dataset_blend_test = np.zeros((len(X_predict), len(clfs)))

    '''5折stacking'''
    n_folds = 5
    skf = list(StratifiedKFold(train_label, n_folds))
    for j, clf in enumerate(clfs):
        '''依次训练各个单模型'''
        # print(j, clf)
        dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
            # print("Fold", i)
            X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
        '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
        print("val acc Score: %f" % accuracy_score(y_predict, dataset_blend_test[:, j]))
    clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
    print("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    print("blend result")
    print("val auc Score: %f" % (roc_auc_score(y_predict, y_submission)))


if __name__ == '__main__':
    batch_size = 128
    resultdir = "./result/temp_result/"
    predict_result(model_name='BiLSTM_Attention',
                   datafile="./modfile/data.pkl",
                   model_file="BiLSTM_Attention.pkl",
                   testfile='./data/mix_data_test_data.json')

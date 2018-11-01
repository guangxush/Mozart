# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-
import sys
import numpy as np
import pickle
import json
import math, codecs
from sklearn.feature_extraction.text import TfidfVectorizer
reload(sys)
sys.setdefaultencoding('utf-8')


def load_vec_txt(fname, vocab, k=100):
    f = codecs.open(fname, 'r', encoding='utf-8')
    w2v = {}
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    unknowtoken = 0
    for line in f:
        if len(line) < k:
            continue
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        w2v[word] = coefs
    f.close()
    w2v["**UNK**"] = np.random.uniform(-1*math.sqrt(3/k), math.sqrt(3/k), k)
    for word in vocab:
        if not w2v.__contains__(word):
            print('UNK---------------- ', word)
            w2v[word] = w2v["**UNK**"]
            unknowtoken +=1
            W[vocab[word]] = w2v[word]
        else:
            W[vocab[word]] = w2v[word]

    print('UnKnown tokens in w2v', unknowtoken)
    return k, W


def load_vec_character(vocab_c_inx, k=30):

    W = np.zeros(shape=(vocab_c_inx.__len__()+1, k))

    for i in vocab_c_inx:
        W[vocab_c_inx[i]] = np.random.uniform(-1*math.sqrt(3/k), math.sqrt(3/k), k)

    return W,k


def load_vec_onehot(vocab_w_inx):
    """
    Loads 300x1 word vecs from word2vec
    """
    k=vocab_w_inx.__len__()
    W = np.zeros(shape=(vocab_w_inx.__len__()+1, k+1))
    for word in vocab_w_inx:
        W[vocab_w_inx[word],vocab_w_inx[word]] = 1.
    # W[1, 1] = 1.
    return k, W


def make_idx_posi_index(file, max_s):
    data_s_all = []
    f = open(file, 'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        p_sent = sent['positions']
        data_p = []

        if len(p_sent) > max_s:
            for i in range(0, max_s):
                list = p_sent[i]
                data_p.append(list)
        else:
            for i in range(len(p_sent)):
                list = p_sent[i]
                data_p.append(list)
            while len(data_p) < max_s:
                list = np.zeros(4)
                data_p.append(list.tolist())

        data_s_all.append(data_p)

    f.close()
    return data_s_all


def make_idx_word_index(file, max_s, max_c, source_vob, target_vob, target_vob_intent, target_vob_slot, source_char, isglobal=True):

    data_s_all = []
    data_t_all = []
    data_c_all = []

    data_t_all_intent = []
    data_t_all_slot = []
    data_t_all_2tag = []

    f = codecs.open(file, 'r', encoding='utf-8')
    fr = f.readlines()
    for num, line in enumerate(fr):
        print(num)
        if len(line) <=1:
            continue
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['words']

        data_t = []
        data_s = []
        if len(s_sent) > max_s:
            i = 0
            while i < max_s:
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["**UNK**"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
        else:
            i = 0
            while i < len(s_sent):
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["**UNK**"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
            num = max_s - len(s_sent)
            for inum in range(0, num):
                data_s.append(0)

        data_s_all.append(data_s)

        if isglobal:
            t_sent = sent['label']
            targetvec = np.zeros(len(target_vob))
            targetvec[target_vob[t_sent]] = 1
            data_t_all.append(targetvec)
        else:
            t_sent_intent = sent['intents']
            targetvec = np.zeros(len(target_vob_intent))
            targetvec[target_vob_intent[t_sent_intent]] = 1
            data_t_all_intent.append(targetvec)

            t_sent_slot = sent['slots']
            targetvec = np.zeros(len(target_vob_slot))
            targetvec[target_vob_slot[t_sent_slot]] = 1
            data_t_all_slot.append(targetvec)

            slot_isNone = sent['isNone']
            if slot_isNone == 'YES':
                data_t_all_2tag.append([0, 1])
            else:
                data_t_all_2tag.append([1, 0])


        data_w = []
        for ii in range(0, min(max_s, len(s_sent))):
            word = s_sent[ii]
            data_c = []
            for chr in range(0, min(word.__len__(), max_c)):
                if not source_char.__contains__(word[chr]):
                    data_c.append(source_char["**UNK**"])
                else:
                    data_c.append(source_char[word[chr]])

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

        data_c_all.append(data_w)

    f.close()
    if isglobal:
        return data_s_all, data_t_all, data_c_all
    else:
        return data_s_all, data_t_all_intent, data_t_all_slot, data_t_all_2tag, data_c_all



def make_idx_pinyin_index(file, max_s, max_c, source_pinyin):

    data_pinyin_all = []


    f = codecs.open(file, 'r', encoding='utf-8')
    fr = f.readlines()
    for num, line in enumerate(fr):
        print(num)
        if len(line) <=1:
            continue
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['words']


        data_w = []
        for ii in range(0, min(max_s, len(s_sent))):
            word = s_sent[ii]
            data_c = []
            for chr in range(0, min(word.__len__(), max_c)):
                if not source_pinyin.__contains__(word[chr]):
                    data_c.append(source_pinyin["**UNK**"])
                else:
                    data_c.append(source_pinyin[word[chr]])

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

        data_pinyin_all.append(data_w)

    f.close()

    return data_pinyin_all



def get_Char_index(files):

    source_vob = {}
    sourc_idex_word = {}
    count = 1
    max_s = 0
    dict = {}
    for file in files:

        f = codecs.open(file,'r',encoding='utf-8')
        fr = f.readlines()
        for line in fr:
            sent = json.loads(line.rstrip('\n').rstrip('\r'))
            sourc = sent['words']
            for word in sourc:
                for i in range(len(word)):
                    if not source_vob.__contains__(word[i]):
                        source_vob[word[i]] = count
                        sourc_idex_word[count] = word[i]
                        count += 1
                if word.__len__() in dict.keys():
                    dict[word.__len__()] = dict[word.__len__()]+1
                else:
                    dict[word.__len__()] = 1
                if word.__len__() > max_s:
                    max_s = word.__len__()
                    # print('max_c  ', max_s, word)

        f.close()

    # for count in dict.keys():
    #     print(count, dict[count])


    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    return source_vob, sourc_idex_word, max_s


def get_Pinyin_index(files):

    source_vob = {}
    sourc_idex_word = {}
    count = 1
    max_s = 0
    dict = {}
    for file in files:

        f = codecs.open(file,'r',encoding='utf-8')
        fr = f.readlines()
        for line in fr:
            sent = json.loads(line.rstrip('\n').rstrip('\r'))
            sourc = sent['words']
            for word_0 in sourc:
                word = word_0.split(' ')
                for i in range(len(word)):
                    if not source_vob.__contains__(word[i]):
                        source_vob[word[i]] = count
                        sourc_idex_word[count] = word[i]
                        count += 1
                if word.__len__() in dict.keys():
                    dict[word.__len__()] = dict[word.__len__()]+1
                else:
                    dict[word.__len__()] = 1
                if word.__len__() > max_s:
                    max_s = word.__len__()
                    # print('max_c  ', max_s, word)

        f.close()

    # for count in dict.keys():
    #     print(count, dict[count])


    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    return source_vob, sourc_idex_word, max_s


def get_Feature_index(file):
    """
    Give each feature labelling an index
    :param the entlabelingfile file
    :return: the word_index map, the index_word map,
    the max lenth of word sentence
    """
    label_vob = {}
    label_idex_word = {}
    count = 1
    # count = 0

    for labelingfile in file:
        f = open(labelingfile, 'r')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:

                continue

            sourc = line.strip('\r\n').rstrip('\n').split(' ')[1]
            # print(sourc)
            if not label_vob.__contains__(sourc):
                label_vob[sourc] = count
                label_idex_word[count] = sourc
                count += 1

        f.close()
    if not label_vob.__contains__("**UNK**"):
        label_vob["**UNK**"] = count
        label_idex_word[count] = "**UNK**"
        count += 1


    return label_vob, label_idex_word


def get_Word_index(files, testfile, isglobal):

    source_vob = {}
    target_vob = {}
    sourc_idex_word = {}
    target_idex_word = {}
    max_s = 0
    tarcount = 0
    count = 1

    target_vob_intent = {}
    target_idex_word_intent = {}
    tarcount_intent = 0
    target_vob_slot = {}
    target_idex_word_slot = {}
    tarcount_slot = 0


    dict = {}
    for file in files:
        f = codecs.open(file, 'r', encoding='utf-8')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:
                continue
            sent = json.loads(line.rstrip('\n').rstrip('\r'))
            sourc = sent['words']
            for word in sourc:
                if not source_vob.__contains__(word):
                    source_vob[word] = count
                    sourc_idex_word[count] = word
                    count += 1

            if sourc.__len__() in dict.keys():
                dict[sourc.__len__()] = dict[sourc.__len__()] + 1
            else:
                dict[sourc.__len__()] = 1

            if sourc.__len__() > max_s:
                max_s = sourc.__len__()
                # print('max_s  ', max_s, sourc)
            if isglobal:
                target = sent['label']

                if not target_vob.__contains__(target):
                    target_vob[target] = tarcount
                    target_idex_word[tarcount] = target
                    tarcount += 1
            else:
                intents = sent['intents']
                slots = sent['slots']

                if not target_vob_intent.__contains__(intents):
                    target_vob_intent[intents] = tarcount_intent
                    target_idex_word_intent[tarcount_intent] = intents
                    tarcount_intent += 1

                if not target_vob_slot.__contains__(slots):
                    target_vob_slot[slots] = tarcount_slot
                    target_idex_word_slot[tarcount_slot] = slots
                    tarcount_slot += 1

        f.close()
    # all = 0
    # for count in sorted(dict.keys()):
    #     all += dict[count]
    #     print(count, dict[count], all)

    if not source_vob.__contains__("**PAD**"):
        source_vob["**PAD**"] = 0
        sourc_idex_word[0] = "**PAD**"

    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    f = codecs.open(testfile, 'r', encoding='utf-8')
    fr = f.readlines()
    for line in fr:
        if line.__len__() <= 1:
            continue
        sent = json.loads(line.rstrip('\n').rstrip('\r'))
        sourc = sent['words']
        for word in sourc:
            if not source_vob.__contains__(word):
                source_vob[word] = count
                sourc_idex_word[count] = word
                count += 1
        if sourc.__len__() > max_s:
            max_s = sourc.__len__()

    f.close()
    if isglobal:
        return source_vob, sourc_idex_word, target_vob, target_idex_word, max_s
    else:
        return source_vob, sourc_idex_word, \
               target_vob_intent, target_idex_word_intent, \
               target_vob_slot, target_idex_word_slot, \
               max_s


def make_idx_char_index(trainfile, max_s, max_c, source_char):
    data_c_all = []

    f1 = codecs.open(trainfile, 'r', encoding='utf-8')
    lines = f1.readlines()
    for num, line in enumerate(lines):
        print(num)
        sent = json.loads(line.rstrip('\n').rstrip('\r'))
        sourc = sent['words']
        data_w = []
        for word in sourc:
            data_c = []
            for chr in range(0, min(word.__len__(), max_c)):
                if not source_char.__contains__(word[chr]):
                    data_c.append(source_char["**UNK**"])
                else:
                    data_c.append(source_char[word[chr]])

            num = max_c - word.__len__()
            for i in range(0, max(num, 0)):
                data_c.append(0)

            data_w.append(data_c)

        num = max_s - len(sourc)
        for inum in range(0, num):
            data_tmp = []
            for i in range(0, max_c):
                data_tmp.append(0)
            data_w.append(data_tmp)

        data_c_all.append(data_w)

    f1.close()

    return data_c_all


def tfidf(trainfile, testfile):

    corpus = []

    f1 = codecs.open(trainfile, 'r', encoding='utf-8')
    lines = f1.readlines()
    for num, line in enumerate(lines):
        sent = json.loads(line.rstrip('\n').rstrip('\r'))
        sourc = ' '.join(sent['words'])
        corpus.append(sourc)
    f1.close()

    f1 = codecs.open(testfile, 'r', encoding='utf-8')
    lines = f1.readlines()
    for num, line in enumerate(lines):
        sent = json.loads(line.rstrip('\n').rstrip('\r'))
        sourc = ' '.join(sent['words'])
        corpus.append(sourc)
    f1.close()

    # 从文件导入停用词表
    stpwrdpath = './data/stop_words.txt'
    stpwrd_dic = codecs.open(stpwrdpath, 'r', encoding='utf-8')
    stpwrd_content = stpwrd_dic.read()
    # 将停用词表转换为list
    stpwrdlst = stpwrd_content.splitlines()
    stpwrd_dic.close()

    vectorizer = TfidfVectorizer(stop_words=stpwrdlst,
                                 analyzer='word',
                                 min_df=2,
                                 token_pattern=r"(?u)\b\w+\b")
    tfidf = vectorizer.fit_transform(corpus)

    weight = tfidf.toarray()

    word = vectorizer.get_feature_names()

    tfidf_train = weight[0:50002]
    print(tfidf_train.shape)
    tfidf_test = weight[50002:]
    print(tfidf_test.shape)

    return tfidf_train, tfidf_test


def get_data(trainfile, testfile, w2v_file, char2v_file, datafile, w2v_k=100, c2v_k=100, maxlen = 50, isglobal = True, left=0):

    char_vob, vob_idex_char, max_c = get_Char_index({trainfile, testfile})
    print("char_vob size: ", char_vob.__len__())
    print("max_c: ", max_c)
    max_c = 6

    # pinyin_vob, vob_idex_pinyin, max_p = get_Pinyin_index({trainfile_pinyin, testfile_pinyin})
    # print("pinyin_vob size: ", pinyin_vob.__len__())
    # print("max_p: ", max_p)
    # max_p = 6
    #
    # pinyin_k, pinyin_W = load_vec_txt(pinyin2v_file, pinyin_vob, k=100)
    # print('pinyin_W shape:', pinyin_W.shape)

    if isglobal:
        word_vob, vob_idex_word, target_vob, vob_idex_target, max_s = get_Word_index({trainfile}, testfile, isglobal)
        print("word_vob vocab size: ", str(len(word_vob)))
        print("max_s: ", max_s)
        print("target vocab size: " + str(target_vob))

        word_k, word_W = load_vec_txt(w2v_file, word_vob, k=w2v_k)
        print("source_W  size: " + str(len(word_W)))
        char_k, char_W = load_vec_txt(char2v_file, char_vob, c2v_k)
        print('char_W shape:', char_W.shape)

        # train_all_char = make_idx_char_index(trainfile, max_s, max_c, char_vob)
        # print('train_all_char size', len(train_all_char))

        train_all, target_all, train_all_char = make_idx_word_index(trainfile, max_s, max_c, word_vob, target_vob,
                                                                    None, None, char_vob,
                                                                    isglobal)
        print('train_all size', len(train_all), 'target_all', len(target_all))
        print('train_all_char size', len(train_all_char))



        train_shuf = train_all
        train_char_shuf = train_all_char
        target_shuf = target_all

        extra_test_num = int(len(train_all) / 5)

        right = left+1
        test = train_shuf[extra_test_num * left:extra_test_num * right]
        test_label = target_shuf[extra_test_num * left:extra_test_num * right]
        train = train_shuf[:extra_test_num * left] + train_shuf[extra_test_num * right:]
        train_label = target_shuf[:extra_test_num * left] + target_shuf[extra_test_num * right:]
        print('extra_test_num', extra_test_num)
        print('train len  ', train.__len__(), len(train_label))
        print('test len  ', test.__len__(), len(test_label))

        test_char = train_char_shuf[extra_test_num * left:extra_test_num * right]
        train_char = train_char_shuf[:extra_test_num * left] + train_char_shuf[extra_test_num * right:]
        print('test_char len  ', test_char.__len__(), )
        print('train_char len  ', train_char.__len__())

        print("dataset created!")
        out = codecs.open(datafile, 'wb')
        pickle.dump([train, train_char, train_label,
                     test, test_char, test_label,
                     word_vob, vob_idex_word, word_W, word_k,
                     target_vob, vob_idex_target,
                     char_vob, vob_idex_char, char_W, char_k,
                     max_s, max_c
                     ], out, 0)
        out.close()


    if not isglobal:
        word_vob, vob_idex_word, target_vob_intent, vob_idex_target_intent, target_vob_slot, vob_idex_target_slot, max_s = get_Word_index({trainfile}, testfile, isglobal)
        print("word_vob vocab size: ", str(len(word_vob)))
        print("max_s: ", max_s)
        print("target vocab size: " + str(vob_idex_target_intent))

        word_k, word_W = load_vec_txt(w2v_file, word_vob, k=w2v_k)
        print("source_W  size: " + str(len(word_W)))
        char_k, char_W = load_vec_txt(char2v_file, char_vob, c2v_k)
        print('char_W shape:',char_W.shape)

        train_all, target_all_intent, target_all_slot, target_all_2tag,  train_all_char = make_idx_word_index(trainfile, max_s, max_c, word_vob, None, target_vob_intent, target_vob_slot, char_vob, isglobal)
        print('train_all size', len(train_all), 'target_all', len(target_all_intent))
        print('train_all_char size', len(train_all_char))

        # train_all_pinyin = make_idx_pinyin_index(trainfile_pinyin, max_s, max_p, pinyin_vob)
        # print('train_all_pinyin size', len(train_all_pinyin))

        train_shuf = train_all
        train_char_shuf = train_all_char
        target_intent_shuf = target_all_intent
        target_slot_shuf = target_all_slot
        target_2tag_shuf = target_all_2tag

        extra_test_num = int(len(train_all) / 5)

        right = left+1
        test = train_shuf[extra_test_num * left:extra_test_num * right]
        test_intent_label = target_intent_shuf[extra_test_num * left:extra_test_num * right]
        test_slot_label = target_slot_shuf[extra_test_num * left:extra_test_num * right]
        test_2tag_label = target_2tag_shuf[extra_test_num * left:extra_test_num * right]
        train = train_shuf[:extra_test_num * left] + train_shuf[extra_test_num * right:]
        train_intent_label = target_intent_shuf[:extra_test_num * left] + target_intent_shuf[extra_test_num * right:]
        train_slot_label = target_slot_shuf[:extra_test_num * left] + target_slot_shuf[extra_test_num * right:]
        train_2tag_label = target_2tag_shuf[:extra_test_num * left] + target_2tag_shuf[extra_test_num * right:]

        print('extra_test_num', extra_test_num)
        print('train len  ', train.__len__(), len(train_intent_label))
        print('test len  ', test.__len__(), len(test_slot_label))

        test_char = train_char_shuf[extra_test_num * left:extra_test_num * right]
        train_char = train_char_shuf[:extra_test_num * left] + train_char_shuf[extra_test_num * right:]
        print('test_char len  ', test_char.__len__(), )
        print('train_char len  ', train_char.__len__())

        # test_pinyin = train_pinyin_shuf[extra_test_num * left:extra_test_num * right]
        # train_pinyin = train_pinyin_shuf[:extra_test_num * left] + train_pinyin_shuf[extra_test_num * right:]
        # print('test_pinyin len  ', test_pinyin.__len__(), )
        # print('train_pinyin len  ', train_pinyin.__len__())

        print ("dataset created!")
        out = codecs.open(datafile, 'wb')
        pickle.dump([train, train_char, train_intent_label, train_slot_label, train_2tag_label,
                     test, test_char, test_intent_label, test_slot_label, test_2tag_label,
                     word_vob, vob_idex_word, word_W, word_k,
                     target_vob_intent, target_vob_slot, vob_idex_target_intent, vob_idex_target_slot,
                     char_vob, vob_idex_char, char_W, char_k,
                     max_s, max_c
                     ], out, 0)
        out.close()


if __name__=="__main__":

    print(20*2)

    maxlen = 50
    trainfile = "../data/intent_train_tagging4train_2.txt"
    testfile = "../data/intent_test_C_tagging4train.txt"
    w2v_file = "../data/CCL_Word2Vec.mod"
    char2v_file = "../data/CCL_Char2Vec.mod"
    w2v_k = 100
    c2v_k = 100
    datafile = "../modfile/data.pkl"
    modelfile = "../modfile/model.pkl"

    char_vob, vob_idex_char, max_c = get_Char_index({trainfile, testfile})
    print("char_vob size: ", char_vob.__len__())
    print("max_c: ", max_c)

    word_vob, vob_idex_word, target_vob, vob_idex_target, max_s = get_Word_index({trainfile}, testfile)
    print("word_vob vocab size: ", str(len(word_vob)))
    print("max_s: ", max_s)
    print("target vocab size: " + str(target_vob))

    word_k, word_W = load_vec_txt(w2v_file, word_vob, k=w2v_k)
    print("source_W  size: " + str(len(word_W)))
    char_k, char_W = load_vec_txt(char2v_file, char_vob, c2v_k)
    print('char_W shape:',char_W.shape)

    # train_all_char = make_idx_char_index(trainfile, max_s, max_c, char_vob)
    # print('train_all_char size', len(train_all_char))

    train_all, target_all, train_all_char = make_idx_word_index(trainfile, max_s, max_c, word_vob, target_vob, char_vob)
    print('train_all size', len(train_all), 'target_all', len(target_all))
    print('train_all_char size', len(train_all_char))
    print(train_all_char[0])

    extra_test_num = int(len(train_all) / 6)

    left = 0
    right = 1
    test = train_all[extra_test_num * left:extra_test_num * right]
    test_label = target_all[extra_test_num * left:extra_test_num * right]
    train = train_all[:extra_test_num * left] + train_all[extra_test_num * right:]
    train_label = target_all[:extra_test_num * left] + target_all[extra_test_num * right:]
    print('extra_test_num', extra_test_num)
    print('train len  ', train.__len__(), len(train_label))
    print('test len  ', test.__len__(), len(test_label))

    test_char = train_all_char[extra_test_num * left:extra_test_num * right]
    train_char = train_all_char[:extra_test_num * left] + train_all_char[extra_test_num * right:]
    print('test_char len  ', test_char.__len__(), )
    print('train_char len  ', train_char.__len__())
    print(train_char[0])
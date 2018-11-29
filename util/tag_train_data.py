# -*- coding:utf-8 -*-

import json, jieba, codecs
from sklearn.feature_extraction.text import TfidfVectorizer


def Tagging(file, fw, istrain):
    fjson = codecs.open(fw, 'w', encoding='utf-8')
    f = codecs.open(file, 'r', encoding='utf-8')
    i = 1
    for line in f.readlines():
        print(i)
        dict = {}
        sent = json.loads(line.strip('\r\n').strip('\n'))
        content0 = sent['content']
        dict['id'] = sent['id']
        # # jieba.load_userdict('./data/jieba_mydict.txt')  # file_name 为文件类对象或自定义词典的路径
        document_cut = jieba.cut(content0)
        result = '@+@'.join(document_cut)
        results = result.split('@+@')
        # # print(result)
        wordlist = []
        for w in results:
            wordlist.append(w)
        dict['words'] = wordlist
        if istrain == True:
            tag = sent['tag']
            dict['label'] = tag
        fj = json.dumps(dict, ensure_ascii=False)
        fjson.write(fj + '\n')
        i += 1
    f.close()
    fjson.close()


def tfidf(trainfile, testfile, tfidf_k=60):
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
    stpwrdpath = '../data/stop_words.txt'
    stpwrd_dic = open(stpwrdpath, 'r')
    stpwrd_content = stpwrd_dic.read()
    # 将停用词表转换为list
    stpwrdlst = stpwrd_content.splitlines()
    stpwrd_dic.close()
    vectorizer = TfidfVectorizer(stop_words=stpwrdlst, analyzer='word', token_pattern=r"(?u)\b\w+\b")
    tfidf = vectorizer.fit_transform(corpus)
    weight = tfidf.toarray()
    word = vectorizer.get_feature_names()
    keywords_tfidf = []
    fjson = codecs.open(path + "keywords_tfidf.txt", 'w', encoding='utf-8')
    for cla in range(len(weight)):
        # train = vectorizer.transform([txtdict[cla]]).toarray()
        keys = []
        wdict = {}
        for j in range(len(word)):
            wdict[word[j]] = weight[cla][j]
        wlist = sorted(wdict.items(), key=lambda x: x[1], reverse=True)
        count = 0
        for id, wtu in enumerate(wlist):
            if wtu[1] > 0.0:
                count += 1
            else:
                break
        if count <= tfidf_k:
            k = count
        else:
            k = (int)(count * 0.5)
        for id, wtu in enumerate(wlist):
            if id >= k:
                break
            if wtu[1] > 0.0:
                keys.append(wtu[0])
            else:
                break
        print(keys)
        # keywords_tfidf.append(wlist[:max(tfidf_k,len(wlist))])
        keywords_tfidf.append(keys)
        dicts = {}
        dicts['keywords_tfidf'] = keys
        fj = json.dumps(dicts, ensure_ascii=False)
        fjson.write(fj + '\n')
    fjson.close()
    print('keywords len---', keywords_tfidf.__len__())
    return keywords_tfidf


if __name__ == '__main__':
    path = '../data/'
    tfidf_k = 100
    trainfile = path + 'mix_data.json'
    fw1 = path + 'mix_data_train_data.json'
    Tagging(trainfile, fw1, istrain=True)
    testfile = path + 'mix_test_data.json'
    fw2 = path + 'mix_data_test_data.json'
    Tagging(trainfile, fw2, istrain=True)
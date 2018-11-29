# coding:utf-8
from gensim.models import word2vec
import jieba
import logging
import codecs
import sys


# 此函数作用是对初始语料进行分词处理后，作为训练模型的语料
def cut_txt(old_files, cut_file, charflag):
    fo = codecs.open(cut_file, 'w', encoding='utf-8')
    fi = codecs.open(old_files, 'r', encoding='utf-8')
    if charflag:
        for text in fi.readlines():
            new_text = text
            # str_out = ' '.join(new_text).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
            #     .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '')\
            #     .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') .replace('’', '')\
            #     .replace('<', '') .replace('>', '')  # 去掉标点符号
            str_out = ' '.join(new_text)
            fo.write(str_out)
    else:
        for text in fi.readlines():
            new_text = jieba.cut(text, cut_all=False)  # 精确模式
            # str_out = ' '.join(new_text).replace('，', '').replace('。', '').replace('？', '').replace('！', '').replace(
            #     '“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '').replace(
            #     '—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '').replace('’', '').replace(
            #     '<', '').replace('>', '')  # 去掉标点符号
            str_out = ' '.join(new_text)
            fo.write(str_out)
    fo.close()
    fi.close()
    return cut_file


def model_train(train_file_name, save_model_file):  # model_file_name为训练语料的路径,save_model为保存模型名
    # 模型训练，生成词向量
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(train_file_name)  # 加载语料
    # 第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5, 第三个参数是神经网络的隐藏层单元数，默认为100
    model = word2vec.Word2Vec(sentences, min_count=1, size=100, window=5, workers=4)
    model.save(save_model_file)
    model.wv.save_word2vec_format(save_model_name, binary=False)   # 以二进制类型保存模型以便重用
    return model


if __name__ == '__main__':
    if sys.argv[1] == 'char':
        save_model_name = '../modfile/Char2Vec.mod'
        cut_file = cut_txt('../data/word2vec.train.data', '../data/jieba_cut_char.txt', charflag=True)
        print("*****cut char finished *******")
        model_1 = model_train(cut_file, save_model_name)
        print("*****char model finished *******")
    elif sys.argv[1] == 'word':
        save_model_name = '../modfile/Word2Vec.mod'
        cut_file = cut_txt('../data/word2vec.train.data', '../data/jieba_cut_word.txt', charflag=False)
        print("*****cut word finished *******")
        model_1 = model_train(cut_file, save_model_name)
        print("*****word model finished *******")
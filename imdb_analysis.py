from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import numpy as np

from model.model1 import lstm_attention_model

pos_list=[]
with open('data/train_pos_all.txt','r',encoding='utf8')as f:
    line=f.readlines()
    pos_list.extend(line)
neg_list=[]
with open('data/train_neg_all.txt','r',encoding='utf8')as f:
    line=f.readlines()
    neg_list.extend(line)
# 创建标签
label=[1 for i in range(12500)]
label.extend([0 for i in range(12499)])
# 评论内容整合
content=pos_list.extend(neg_list)
content=pos_list


# 去掉停用词和标点符号
seq=[]
seqtence=[]
# nltk.download("stopwords")
# nltk.download("punkt")
stop_words=set(stopwords.words('english'))
for con in content:
    words=nltk.word_tokenize(con)
    line=[]
    for word in words:
        if word.isalpha() and word not in stop_words:
            line.append(word)
    seq.append(line)
    seqtence.extend(line)
print(len(seqtence))


# 获取词索引
tokenizer = Tokenizer()
tokenizer.fit_on_texts(content)
one_hot_results = tokenizer.texts_to_matrix(content, mode='binary')
word_index = tokenizer.word_index
word_index=tokenizer.word_index
sequences=tokenizer.texts_to_sequences(seq)
# 此处设置每个句子最长不超过 800
final_sequences=sequence.pad_sequences(sequences,maxlen=800)


# 转换为numpy类型
label=np.array(label)
# 随机打乱数据
indices=np.random.permutation(len(final_sequences)-1)
X=final_sequences[indices]
y=label[indices]
# 划分测试集和训练集
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2)


# 网络构建
model = lstm_attention_model(input_dim=800, sourcevocabsize=89483, output_dim=1)
model.fit(Xtrain,ytrain,batch_size=32,epochs=10,validation_data=(Xtest,ytest))

# https://www.jianshu.com/p/6b16b592b08d





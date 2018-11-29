# -*- encoding:utf-8 -*-

from model.model2 import mlp2
from util.data_load import load_data3
import numpy as np
from util.util import cal_err_ratio
from util.data_load import generate_model2_data


# use this model test model1&model2 or generate the result
def model_use(i):
    data_file = "./modfile/data_fold_"
    filepath = "./modfile/model2file/mlp.best_model.h5"
    result_path = './data/test_model2_data/news_'+str(i)+'.data'
    model_name = 'BiLSTM_Attention'
    modle_file = "BiLSTM_Attention_fold_"
    testfile = './data/mix_data_test_data.json'
    batch_size = 128
    generate_model2_data(model_name=model_name, datafile=data_file, model_file=modle_file, testfile=testfile,
                         result_path=result_path, batch_size=batch_size)
    x_test, y_test = load_data3(data_path=result_path)
    model2 = mlp2(sample_dim=x_test.shape[1], class_count=1)
    model2.load_weights(filepath)
    results = model2.predict(x_test)
    label = np.argmax(results, axis=1).astype('int')
    print("pred:")
    print(label)
    print("true:")
    print(y_test)
    cal_err_ratio(file_name='test', label=label, y_test=y_test)


if __name__ == '__main__':
    for i in range(1, 6):
        model_use(i=i)
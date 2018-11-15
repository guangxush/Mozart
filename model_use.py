# -*- encoding:utf-8 -*-

from model.model2 import mlp2
from util.data_load import load_data3
import numpy as np
from util.util import cal_err_ratio
from util.data_process import generate_model2_data


# use this model test model1&model2 or generate the result
def model_use(i):
    data_path = './data/test_data/mnist_'+str(i)+'.data'
    filepath = "./modfile/model2file/mlp.best_model.h5"
    result_path = './data/test_model2_data/mnist_'+str(i)+'.data'
    num_classes = 10
    generate_model2_data(data_path=data_path, result_path=result_path, num_classes=num_classes)
    x_test, y_test = load_data3(data_path=result_path)
    mlp_model = mlp2(sample_dim=x_test.shape[1], class_count=10)
    mlp_model.load_weights(filepath)
    results = mlp_model.predict(x_test)
    label = np.argmax(results, axis=1).astype('int')
    print("pred:")
    print(label)
    print("true:")
    print(y_test)
    cal_err_ratio(file_name='test', label=label, y_test=y_test)


if __name__ == '__main__':
    for i in range(1, 6):
        model_use(i=i)
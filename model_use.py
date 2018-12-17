# -*- encoding:utf-8 -*-
from sklearn.externals import joblib

from util.data_load import load_data3
from util.util import cal_err_ratio
from util.data_load import generate_imdb_model2_data


# use this model test model1&model2 or generate the result
def model_use(i):
    filepath = "./modfile/model2file/imdb.xgb.best_model.pkl"
    result_path = './data/model2_result/imdb_' + str(i) + '_data.csv'
    model_file = './modfile/model1file/lstm.best_model_'
    test_file = './data/part_data_all/test_1.txt'
    generate_imdb_model2_data(model_file=model_file, result_path=result_path,
                              test_file=test_file, count=10)
    print('Load result ...')
    x_test, y_test = load_data3(data_path=result_path)
    model2_xgb = joblib.load(filepath)
    results = model2_xgb.predict(x_test)
    label = results
    cal_err_ratio(file_name='test', label=label, y_test=y_test)


if __name__ == '__main__':
    for i in range(1, 10):
        model_use(i=i)
# -*- encoding:utf-8 -*-
import os, time, codecs


# auto delete the unuseful file: h5files, result files...
def delete_unuseful_file(files):
    if not os.path.exists(files):
        print(files)
        print("file not found!")
        return
    else:
        for name in files:
            print(name)
            os.remove(os.path.join(files, name))
            print(name+'delete!')


# delete files in floder
def delete_files():
    files = '../modfile/model1file/'
    delete_unuseful_file(files)


# save logs in training
def print_log(logs):
    t = str(int(time.time()))
    fw = codecs.open("./logs/" + "classify_logs_" + t[0:8] + ".txt", 'a', encoding='utf-8')
    fw.write(logs+"\n")
    fw.close()


# calculate the error ratio of model
def cal_err_ratio(file_name, label, y_test):
    err_count = 0
    sum_count = 0
    t = str(int(time.time()))
    fw = codecs.open("./result/" + file_name + "_classify_result_" + t[0:8] + ".txt", 'a', encoding='utf-8')
    for i in label:
        if i != y_test[sum_count]:
            err_count += 1
        sum_count += 1
    err_ratio = float(err_count) / float(sum_count)
    print(err_count)
    print(sum_count)
    print(err_ratio)
    fw.write("pred_result:"+str(label)+'\n')
    fw.write("true_result:"+str(y_test)+'\n')
    fw.write("err_ratio:"+str(err_ratio)+'\n')
    fw.close()


if __name__ == '__main__':
    delete_files()
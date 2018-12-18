# -*- encoding:utf-8 -*-


# imdb genarate train data
def generate_imdb_train_data(in_pos_file, in_neg_file, part, count):
    for i in range(part):
        out_all_file = "../data/part_data_all/train_" + str(i+1) + ".txt"
        fw = open(out_all_file, 'w', encoding='utf-8')
        with open(in_pos_file, 'r', encoding='utf8')as f:
            pos_line = f.readlines()[count * i:count * (i + 1)]
        with open(in_neg_file, 'r', encoding='utf8')as f:
            neg_line = f.readlines()[count * i:count * (i + 1)]
        for k in range(len(pos_line)):
            fw.write(pos_line[k].rstrip('\n') + '@@@1\n')
            fw.write(neg_line[k].rstrip('\n') + '@@@0\n')
        print("data " + str(i+1) + " processed!")
    return


# imdb generate test data
def generate_imdb_test_data(in_pos_file, in_neg_file, start_line, part, count):
    for i in range(part):
        out_all_file = "../data/part_data_all/test_" + str(i) + ".txt"
        fw = open(out_all_file, 'w', encoding='utf-8')
        with open(in_pos_file, 'r', encoding='utf8')as f:
            pos_line = f.readlines()[start_line + count * i:start_line + count * (i + 1)]
        with open(in_neg_file, 'r', encoding='utf8')as f:
            neg_line = f.readlines()[start_line + count * i:start_line + count * (i + 1)]
        for k in range(len(neg_line)):
            fw.write(pos_line[k].rstrip('\n') + '@@@1\n')
            fw.write(neg_line[k].rstrip('\n') + '@@@0\n')
        print("data " + str(i) + " processed!")
    return


# imdb genarate train data
def generate_imdb_all_train_data(in_pos_file, in_neg_file, part, count):
    for i in range(part):
        out_all_file = "../data/part_data_all/all_train.txt"
        fw = open(out_all_file, 'w', encoding='utf-8')
        with open(in_pos_file, 'r', encoding='utf8')as f:
            pos_line = f.readlines()[count * i:count * (i + 1)]
        with open(in_neg_file, 'r', encoding='utf8')as f:
            neg_line = f.readlines()[count * i:count * (i + 1)]
        for k in range(len(pos_line)):
            fw.write(pos_line[k].rstrip('\n') + '@@@1\n')
            fw.write(neg_line[k].rstrip('\n') + '@@@0\n')
        print("data " + str(i) + " processed!")
    return


# imdb generate test data
def generate_imdb_all_test_data(in_pos_file, in_neg_file, start_line, part, count):
    for i in range(part):
        out_all_file = "../data/part_data_all/all_test.txt"
        fw = open(out_all_file, 'w', encoding='utf-8')
        with open(in_pos_file, 'r', encoding='utf8')as f:
            pos_line = f.readlines()[start_line + count * i:start_line + count * (i + 1)]
        with open(in_neg_file, 'r', encoding='utf8')as f:
            neg_line = f.readlines()[start_line + count * i:start_line + count * (i + 1)]
        for k in range(len(neg_line)):
            fw.write(pos_line[k].rstrip('\n') + '@@@1\n')
            fw.write(neg_line[k].rstrip('\n') + '@@@0\n')
        print("data " + str(i) + " processed!")
    return


if __name__ == '__main__':
    # generate_imdb_train_data(in_pos_file='../data/train_pos_all.txt', in_neg_file='../data/train_neg_all.txt',
    #                          part=10, count=150)
    # generate_imdb_test_data(in_pos_file='../data/train_pos_all.txt', in_neg_file='../data/train_neg_all.txt',
    #                         start_line=1500, part=2, count=50)
    generate_imdb_all_train_data(in_pos_file='../data/train_pos_all.txt', in_neg_file='../data/train_neg_all.txt', part=1,
                                 count=1500)
    generate_imdb_all_test_data(in_pos_file='../data/train_pos_all.txt', in_neg_file='../data/train_neg_all.txt',
                                start_line=1500, part=1, count=50)

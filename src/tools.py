#! -*- coding:utf-8 -*-


import pandas as pd


def load_cls_data(filename):
    '''
        加载分类数据
    :param filename: path
    :return:
    '''
    D = []
    df = pd.read_csv(filename, encoding='utf-8')

    files = df['0']
    tags = df['1']

    for f, t in zip(files, tags):
        D.append((f, int(t)))
    return D


def load_reg_data(filename):
    '''
        加载回归数据
    :param filename: path
    :return:
    '''
    D = []
    df = pd.read_csv(filename, encoding='utf-8')

    files = df['0']
    tags = df['1']

    n = 0
    for f, t in zip(files, tags):
        D.append((f, t))
        # n += 1
        # if n == 100:
        #     break

    return D


def write_to_csv(data_list, dst_path, col=None):
    '''
    如果不传入 col, 则默认输出列名为 0， 1， 2，...;传入则输出传入的列名。
    '''
    if col:
        csv_data = pd.DataFrame(data_list, columns=col)
        csv_data.to_csv(dst_path, index=False, encoding='utf_8_sig')  # 此处要加 index=False，这样的话，就去掉了数字索引。
    else:
        csv_data = pd.DataFrame(data_list)
        csv_data.to_csv(dst_path, encoding='utf_8_sig')  # 默认列是 数字索引。

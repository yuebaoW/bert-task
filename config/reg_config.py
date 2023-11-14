#! -*- coding:utf-8 -*-


import os
from datetime import date
today = date.today()


class Train_parameters:
    the_time = today

    GPU = '1'

    maxlen = 512  # 输入模型最大长度  原始 350, 最大 512
    layer = 12
    pre_train_root_dir = '/workspace/model/chinese_roberta_wwm_ext_L-12_H-768_A-12/'  # 原装
    # layer = 6
    # pre_train_root_dir = '/workspace/model/roberta_zh_L-6-H-768_A-12/'  # 6层

    batch = 32
    epoch = 20
    train_data_root = './data/1108_reg'

    # 保存权重
    checkpoints_root = './reg_checkpoints/bert_{}_{}'.format(layer, the_time)
    if not os.path.exists(checkpoints_root):
        os.mkdir(checkpoints_root)
    save_model_name = 'best_model_{}.weights'.format(maxlen)
    checkpoints_path = os.path.join(checkpoints_root, save_model_name)


class To_h5_parameters:
    '''
        转 h5 模型所需的参数, 除了参数 load_model_time_2_h5， 其他都不用动
    '''
    model_p = Train_parameters()

    load_model_time_2_h5 = '2023-11-13'

    load_checkpoints_root = './reg_checkpoints/bert_{}_{}'.format(model_p.layer, load_model_time_2_h5)
    load_checkpoints_path = os.path.join(load_checkpoints_root, model_p.save_model_name)
    # 转 h5 命名，要和训练保存的名字已知，避免混乱。
    save_h5_path = './reg_checkpoints/regression-bert_{}.h5'.format(load_model_time_2_h5)


class Infer_parameters():
    '''
        加载的模型路径。
    '''
    modul_time = '2023-11-13'
    h5_path = './reg_checkpoints/regression-bert_{}.h5'.format(modul_time)


train_parameters = Train_parameters()
to_h5_parameters = To_h5_parameters()
infer_parameters = Infer_parameters()

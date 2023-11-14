#! -*- coding:utf-8 -*-

'''

    Bert 分类

'''


import sys
sys.path.append("..")

from config.cls_config import train_parameters

from bert4keras.backend import keras, set_gelu
from bert4keras.models import build_transformer_model
from keras.layers import Lambda, Dense


set_gelu('tanh')  # 切换gelu版本


pre_train_root_dir = train_parameters.pre_train_root_dir
config_path = pre_train_root_dir + 'bert_config.json'  # 配置文件
pre_train_path = pre_train_root_dir + 'bert_model.ckpt'  # 预训练模型


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=pre_train_path,
    model='bert',
    return_keras_model=False,
)

# 做回归任务和bert一样，都可以取第一个token对应的编码，做成一个序列，然后去判断。
output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

output = Dense(
    units=train_parameters.class_num,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)



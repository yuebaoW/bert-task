#! -*- coding:utf-8 -*-

'''

    Bert 回归模型

分类任务改成回归：
1、输出维度变为1；
2、是否用sigmoid，看情况，用的话，好处：输出值 0<y<1;略处：若拟合高斯分布，将会导致波峰更加细长。
    在我们的评分系统中，建议不实用激活。
3、使用激活确实会稳定，相比不实用，会让全局绝对误差更小。
'''


import sys
sys.path.append("..")

from config.reg_config import train_parameters

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
    units=1,  # 输出维度，因为是回归，所以=1即可；softmax要去掉，不然会出现数值异常
    # activation='sigmoid',  # 回归任务加这个函数，是为了让输出值 >0 和 <1。目前看，该函数最合适。
    kernel_initializer=bert.initializer
    )(output)

model = keras.models.Model(bert.model.input, output)



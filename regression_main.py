#! -*- coding:utf-8 -*-

import sys
import os
sys.path.append("..")

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

os.environ['TF_KERAS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import tensorflow as tf

from src.tools import load_reg_data, write_to_csv
from src.evaluate import Evaluator_reg

from modules.data_generator import data_generator, tokenizer
from modules.reg_model import model

from config.reg_config import train_parameters, to_h5_parameters, infer_parameters
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr


data_root = train_parameters.train_data_root
test_data_path = data_root + '/test.csv'
train_data_path = data_root + '/train.csv'

test_data = load_reg_data(test_data_path)
train_data = load_reg_data(train_data_path)

# 转换数据集
train_generator = data_generator(train_data, train_parameters.batch)
valid_generator = data_generator(test_data, train_parameters.batch)
test_generator = data_generator(test_data, train_parameters.batch)


# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(
    loss='MSE',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
        1000: 1,
        2000: 0.1
    }),
    metrics=['mse'],
)


def train():
    evaluator = Evaluator_reg(model, valid_generator, train_parameters.checkpoints_path)
    model.fit(
        train_generator.forfit(),  # 这里传数据，开始跑了，然后测试和验证。
        steps_per_epoch=len(train_generator),
        epochs=train_parameters.batch,
        callbacks=[evaluator]
    )


def model_2_h5():
    '''
        转 h5
    :return:
    '''
    print('0000', to_h5_parameters.load_checkpoints_path)
    model.load_weights(to_h5_parameters.load_checkpoints_path)
    model.summary()
    # 保存 h5 的形式，可以直接tf.keras.models.load_model加载。就是保存了所有的参数，不再需要任何模型配置了。
    model.save(to_h5_parameters.save_h5_path, overwrite=True, include_optimizer=False)
    print('转 pb 代码成功。')


def predict():
    model = '2023-11-06'

    col = ['text', 'GT', 'preb']

    # 转换数据集
    mod = tf.keras.models.load_model(infer_parameters.h5_path)

    mean_abs_ls = []  # 对误差取绝对值
    output = []
    for (text, tag) in test_data:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=train_parameters.maxlen)
        X = [np.array([token_ids]), np.array([segment_ids])]
        Y = mod.predict(X)
        y_preb = Y[0]
        print(text, 'GT:', tag, 'infor:', y_preb)

        output.append([text, tag, y_preb])

        try:
            mean_abs_ls.append(abs(tag - y_preb[0]))
        except:
            continue
    print('全局绝对误差均值', np.mean(mean_abs_ls))

    output.append(['全局绝对误差均值', np.mean(mean_abs_ls), ''])

    write_to_csv(output, 'output/{}.csv'.format(model), col)


def main():
    # train()
    # model_2_h5()
    predict()


if __name__ == '__main__':
    main()

#! -*- coding:utf-8 -*-

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, set_gelu
from sklearn.metrics import classification_report


def evaluate_reg(model, data):
    '''
        计算均方误差的均值。
    :param data:
    :return:
    '''
    out_ls = []
    for x_true, y_true in data:
        y_pred = model.predict(x_true)

        # print('真：', y_true)
        # print('推理：', y_pred)

        mse = tf.keras.losses.MeanSquaredError()
        out = mse(y_true, y_pred).numpy()
        # print('均方差：', out)
        out_ls.append(out)

    return np.mean(out_ls)


class Evaluator_reg(keras.callbacks.Callback):
    """回归模型的 评估与保存
    """
    def __init__(self, model, val_data, checkpoints_path):
        self.best_val_acc = 0.1  # 均方差的均值 <0.1, 就开始保存模型。
        self.model = model
        self.val_data = val_data
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate_reg(self.model, self.val_data)
        if val_acc < self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.checkpoints_path)
            print('Saved_model!')
        print('global_error:', val_acc)


def evaluate_cls(model, data):
    yt = []
    yp = []
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
        yt.extend(y_true.tolist())
        yp.extend(y_pred.tolist())
    print(classification_report(yt, yp))
    return right / total


class Evaluator_cls(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, model, val_data, checkpoints_path):
        self.best_val_acc = 0.
        self.model = model
        self.val_data = val_data
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate_cls(self.model, self.val_data)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.checkpoints_path)
        test_acc = evaluate_cls(self.model, self.val_data)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )

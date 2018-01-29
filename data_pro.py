# encoding:utf-8
import numpy as np
import random
# 读取文件
def load_data(file):
    buff = []
    with open(file, 'r') as f_r:
        for line in f_r:
            line = line.strip().decode('utf-8').split()
            buff.append(line)
    return buff
# 读取数据


# 构建字典，key为词语，value为编号
def build_dict(buff, _dict):
    for sent in buff:
        for w in sent:
            if w not in _dict:
                _dict[w] = len(_dict)
    return _dict


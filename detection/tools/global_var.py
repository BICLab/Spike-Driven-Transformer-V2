# -*- coding: utf-8 -*-

def init():  # 初始化
    global _global_dict
    _global_dict = {}

def set_value(key, value):
    #定义一个全局变量
    global _global_dict
    _global_dict[key] = value

def get_value(key):
    #获得一个全局变量，不存在则提示读取对应变量失败
    global _global_dict
    try:
        return _global_dict[key]
    except:
        print('读取'+key+'失败\r\n')

def change_value(key,delta):
    #获得一个全局变量，不存在则提示读取对应变量失败,存在则对其增加delta
    global _global_dict
    try:
        _global_dict[key] = _global_dict[key] + delta #如果变量存在则把delta加上
        return _global_dict[key]
    except:
        _global_dict[key] = delta  #否则直接将变量set上


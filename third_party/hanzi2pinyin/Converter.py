# coding:utf8
"""
@author basicworld@163.com
"""
import os
import sys
import importlib

importlib.reload(sys)

import re
BASEDIR = os.path.dirname(os.path.abspath(__file__))
DICT_FILENAME = os.path.join(BASEDIR, 'pinyin_dict.txt')

class Converter(object):
    def __init__(self, dict_filename=DICT_FILENAME, jointer=',',
                 mark_tone=True, char_if_not_match='XX0',
                 ignore_num=True, ignore_alphabet=True):
        """
        汉字转化为拼音
        @dict_filename 词典库文件
        @jointer 字符串连接符
        @mark_tone 是否输出声调
        @char_if_not_match 未匹配时的替代值
        @ignore_num 忽略数字  # todo 功能未完成
        @ignore_alphabet 忽略英文字母  # todo 功能未完成

        e.g.
        >>c = Converter()
        >>print c.convert('中国')
        """
        self._dict = {}
        for line in open(dict_filename, encoding='utf-8'):
            line = line.rstrip()
            self._dict[line[0]] = line[1:].split(',')

        self._jointer = jointer
        self._mark_tone = mark_tone
        self._char_if_not_match = char_if_not_match

    def convert(self, string):
        """
        @string unicode或者utf8编码的汉字字符串
        """
        #string = string.decode("utf-8")
        pinyin = []
        # map 层
        for key in string:
            pinyin.append(self._dict.get(key, [self._char_if_not_match])[0])

        # 输出层
        if self._mark_tone:
            return self._jointer.join(pinyin)
        else:
            return self._jointer.join([i[:-1] for i in pinyin])

    def __del__(self):
        pass


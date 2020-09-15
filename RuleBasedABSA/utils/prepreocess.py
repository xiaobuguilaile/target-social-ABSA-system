#!/anaconda3/envs/nlp python3.8
# -*- coding: utf-8 -*-
# ---
# @File: prepreocess.py
# @Author: HW Shen
# @Time: 9月 15, 2020
# ---
import jieba_fast as jieba

STOPWORDS = [""]


def tokenize(text, isUseStopwords):

    if isUseStopwords:
        words = [word for word in jieba.lcut(text) if word not in STOPWORDS]
    else:
        words = [word for word in jieba.lcut(text)]

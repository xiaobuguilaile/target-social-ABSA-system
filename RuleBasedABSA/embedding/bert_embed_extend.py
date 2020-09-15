#!/anaconda3/envs/nlp python3.8
# -*- coding: utf-8 -*-
# ---
# @File: bert_embed_extend.py
# @Author: HW Shen
# @Time: 9月 15, 2020
# ---

from bert_embedding import BertEmbedding
import mxnet as mx
import numpy as np
from RuleBasedABSA.utils.prepreocess import tokenize


class BertEmbeddingExtend(object):

    def __init__(self):

        self.bert_embed = BertEmbedding(model='bert_12_768_12',
                                        dataset_name="wiki_cn")

    def getSentenceEmbedding(self, sentence, isUseAveragePooling, isUseStopwords):

        if isUseStopwords:
            words = tokenize(sentence, isUseStopwords)
            if len(words) == 0:
                return np.zeros(768)
            sentence = " ".join(words)

        result = self.bert_embed(sentences=sentence.split("\n"))
        first_sentence = result[0]

        if first_sentence[1] == None or len(first_sentence[1]) == 1:
            return np.zeros(768)

        w_v = np.array(first_sentence[1])
        total_effect_count = w_v.shape[0]

        if isUseAveragePooling:
            w_v = np.sum(w_v, axis=0) / total_effect_count
        else:
            w_v = np.max(w_v, axis=0)

        return w_v


if __name__ == '__main__':

    bertem = BertEmbeddingExtend()
    print(bertem.getSentenceEmbedding("包装非常满意，但是价格太贵了", False, False))
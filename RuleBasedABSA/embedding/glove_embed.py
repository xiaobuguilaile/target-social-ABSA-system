#!/anaconda3/envs/nlp python3.8
# -*- coding: utf-8 -*-
# ---
# @File: glove_embed.py
# @Author: HW Shen
# @Time: 9月 15, 2020
# ---

import numpy as np
from RuleBasedABSA.utils.prepreocess import tokenize
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(BASE_DIR)


class GloveEmbedding(object):

    def __init__(self):
        self.embeddings_index = {}
        self.embeddings_dim_glove = 100  # glove dimension
        self.init_data()

    def init_data(self):

        glovefile = open(BASE_DIR + "/data/glove.6B.100d.txt", encoding="utf-8")

        for line in glovefile:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float16')
            self.embeddings_index[word] = coefs
        glovefile.close()

    def get_embedding_matrix_glove(self, word):

        embedding_vector = self.embeddings_index.get(word)
        if embedding_vector is not None:
            return embedding_vector[:self.embeddings_dim_glove]
        return np.zeros(self.embeddings_dim_glove)

    def getSentenceVectorCommon(self, sentence, isUseAveragePooling, isUseStopwords):

        tokens = tokenize(sentence, isUseStopwords)

        total_effect_count = 0
        w_v = []
        for word in tokens:
            if word in self.embeddings_index:
                total_effect_count += 1
                w_v.append(self.embeddings_index[word])

        w_v = np.array(w_v)

        is_effect = total_effect_count > 0
        if is_effect:
            if isUseAveragePooling:
                w_v = np.sum(w_v, axis=0) / total_effect_count
            else:
                w_v = np.max(w_v, axis=0)
        else:
            w_v = np.zeros(self.embeddings_dim_glove)

        return np.array(w_v)


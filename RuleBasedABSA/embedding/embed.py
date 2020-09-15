#!/anaconda3/envs/nlp python3.8
# -*- coding: utf-8 -*-
# ---
# @File: embed.py
# @Author: HW Shen
# @Time: 9月 15, 2020
# ---

from RuleBasedABSA.embedding.glove_embed import GloveEmbedding
from RuleBasedABSA.embedding.bert_embed_extend import BertEmbeddingExtend
from enum import Enum


class EmbeddingType(Enum):

    glove = 0
    bert = 1


class EmbedManager(object):

    def __init__(self):

        self.g_embed = GloveEmbedding()
        self.b_embed = BertEmbeddingExtend()

    def getEmbedding(self, sentence, type, isUseAveragePooling, isUseStopwords):
        if type == EmbeddingType.glove:
            return


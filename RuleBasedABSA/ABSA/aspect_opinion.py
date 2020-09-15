#!/anaconda3/envs/nlp python3.8
# -*- coding: utf-8 -*-
# ---
# @File: aspect_opinion.py
# @Author: HW Shen
# @Time: Sep 15, 2020
# ---

import json
import pandas as pd
import jieba_fast as jieba
import jieba_fast.posseg as psg
import stanza


class AspectOpinion(object):

    def __init__(self):

        self.stanz_nlp = stanza.Pipeline(lang='zh')

    def get_segment(self, text, aspect):
        """ Split text into ited_txts accodring to aspect """

        if self.is_text_only_one_aspect(text):
            return text

        cur_aspect_index = text.index(aspect)
        cur_aspect_end_index_begin = cur_aspect_index + len(aspect)  # first_index after aspect
        cur_aspect_end_index_end = cur_aspect_end_index_begin
        end_pos = len(text) - 1

        stop_punct_map = {c: None for c in "，。！？；"}  # punctuation
        relation_punct_list = ["但是", "但", "却","然而", "可是", "只是", "不过", "偏偏", "可惜"]

        cur_aspect_des = self.get_cur_aspect_adj(text[cur_aspect_end_index_begin:end_pos])

        while cur_aspect_end_index_end <= end_pos:
            # punctuation
            cur_str = text[cur_aspect_end_index_end:min(cur_aspect_end_index_end+1, end_pos)]
            if cur_str in stop_punct_map:
                break

            #
            cur_strs = text[cur_aspect_end_index_begin:cur_aspect_end_index_end]
            relation_store = ""
            for relation in relation_punct_list:
                if relation in cur_str:
                    relation_store = relation
                    break

            if relation_store != "":
                cur_aspect_end_index_end -= len(relation_store)
                break

            if cur_aspect_des != None:
                if cur_aspect_des in cur_strs:
                    break
            cur_aspect_end_index_end += 1

        cur_aspect_end_index_end = min(cur_aspect_end_index_end, end_pos)

        return text[cur_aspect_index: cur_aspect_end_index_end]

    def get_cur_aspect_adj(self, seg_text):
        """ Find out adj as opinion for aspect """
        doc = self.stanz_nlp(seg_text)
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos == "ADJ":
                    return word
        return None

    def is_text_only_one_aspect(self, text):
        """ text only contains one aspect """
        tagged_words = []
        doc = self.stanz_nlp(text)
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos == "NOUN":
                    tagged_words.append(word.text)
        if len(tagged_words) <= 1:
            return True

        return False

    def extract_aspects(self, ):
        pass


if __name__ == '__main__':
    ao = AspectOpinion()
    text = "包装非常满意，但是价格太贵了"
    res = ao.get_segment(text,aspect="包装")
    print(res)

    # stanz_nlp = stanza.Pipeline('zh')
    # doc = stanz_nlp(text)
    # for sent in doc.sentences:
    #     print("UPOS: " + ' '.join(f'{word.text}/{word.upos}' for word in sent.words))  # 词性标注（UPOS）

    # for (word, tag) in psg.lcut(text):
    #     print(word, tag)

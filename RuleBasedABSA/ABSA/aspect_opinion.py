#!/anaconda3/envs/nlp python3.8
# -*- coding: utf-8 -*-
# ---
# @File: aspect_opinion.py
# @Author: HW Shen
# @Time: Sep 15, 2020
# ---

import json
import pandas as pd
import stanza
from RuleBasedABSA.utils.prepreocess import tokenize


class AspectOpinion(object):

    def __init__(self):

        self.stanze_nlp = stanza.Pipeline(lang='zh')
        self.aspect_filter = []

    def get_segment(self, text, aspect):
        """ 通过标点符号和转折词对 text 进行切分，获取某个 aspect 的短句 """

        if self.is_text_only_one_aspect(text):
            return text

        cur_aspect_index = text.index(aspect)
        cur_aspect_end_index_begin = cur_aspect_index + len(aspect)  # first_index after aspect
        cur_aspect_end_index_end = cur_aspect_end_index_begin
        end_pos = len(text) - 1

        stop_punct_map = {c: None for c in "，。！？；"}  # punctuation
        relation_punct_list = [" ", "但是", "但", "却", "然而", "可是", "只是", "不过", "偏偏", "可惜"]

        cur_aspect_des = self.get_cur_aspect_adj(text[cur_aspect_end_index_begin:end_pos])

        while cur_aspect_end_index_end <= end_pos:
            # 在 “标点符号” 处截取
            cur_str = text[cur_aspect_end_index_end:min(cur_aspect_end_index_end+1, end_pos)]
            if cur_str in stop_punct_map:
                break

            # 在 “转移词” (eg. 但是, 却, 然而...)处截取
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
                # print("cur_aspect_des: ", cur_aspect_des)
                if cur_aspect_des in cur_strs:
                    break
            cur_aspect_end_index_end += 1

        cur_aspect_end_index_end = min(cur_aspect_end_index_end, end_pos)

        return text[cur_aspect_index: cur_aspect_end_index_end], text[cur_aspect_end_index_end:]

    def get_cur_aspect_adj(self, seg_text):
        """ 从text片段中查找 形容词 """
        doc = self.stanze_nlp(seg_text)
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos == "ADJ":
                    return word.text
        return None

    def is_text_only_one_aspect(self, text):
        """ 判断评论里面是否只包含一个方面 """
        tagged_words = []
        doc = self.stanze_nlp(text)
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos == "NOUN":
                    tagged_words.append(word.text)
        if len(tagged_words) <= 1:
            return True

        return False

    def extract_aspects(self, text_list):
        """
        从 text 中抽取所有的 aspects
        思路是提取句子中的名词(英文中用NN表示)，然后根据出现的次数排序，取出现次数做多的前5个
        """
        aspects_dic = {}
        for sentence in text_list:
            if sentence == None or str.strip(sentence) == "":
                continue
            doc = self.stanze_nlp(sentence)
            # print("doc: ", doc)
            for sent in doc.sentences:
                for word in sent.words:
                    if word.upos == "NOUN":
                        if word.text not in aspects_dic:
                            aspects_dic[word.text] = []
                        aspects_dic[word.text].append(sentence)
        print("aspects_dic: ", aspects_dic)
        aspects_sorted = sorted(aspects_dic.items(), key=lambda x:len(x[1]), reverse=True)
        print("aspects_sorted: ", aspects_sorted)
        aspects_dic = {}
        for index, item in enumerate(aspects_sorted):
            # 一般情况下，我们提取出来的 aspect由于都是出现次数比较多的，很符合aspect的特性，如food、service、location等名词，
            # 但是也有特殊情况, 如很多用户可能评价酒店的老板 "the boss is a good man", 显然 man不能作为 aspect，用这个过滤器把它过滤掉
            if item[0] in self.aspect_filter:
                continue

            if len(aspects_dic.items()) < 5:
                aspects_dic[item[0]] = item[1]
        print("aspects_dic: ", aspects_dic)

        return aspects_dic


if __name__ == '__main__':

    a_o = AspectOpinion()

    # text = ["针对包装非常满意", "但是价格太贵了", "包装太满意", "但价格贵了", "包装满意"]
    text = "针对包装非常满意但是价格太贵了"
    res, left = a_o.get_segment(text, aspect="包装")
    print(res)
    print(left)


    # a_o.extract_aspects(text)
    # stanz_nlp = stanza.Pipeline('zh')
    # doc = stanz_nlp(text)
    # for sent in doc.sentences:
    #     print("UPOS: " + ' '.join(f'{word.text}/{word.upos}' for word in sent.words))  # 词性标注（UPOS）

    # for (word, tag) in psg.lcut(text):
    #     print(word, tag)

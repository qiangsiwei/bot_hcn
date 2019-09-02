# -*- coding: utf-8 -*-

import os
import numpy as np
from gensim.models import word2vec

class NLU(object):

    def __init__(self, voc_fn, w2v_fn, w2v_dim, entity_dict):
        self.voc = self.get_voc(voc_fn)
        self.voc_dim = len(self.voc)
        self.w2v = word2vec.Word2Vec.load(w2v_fn)
        self.w2v_dim = w2v_dim
        self.entity_dict = entity_dict

    def get_voc(self, voc_fn):
        with open(voc_fn) as fin:
            voc = fin.read().split('\n')
        return voc

    def get_bow_vector(self, text):
        vec = np.zeros([self.voc_dim],dtype=np.int32)
        for word in text.split(' '):
            if word in self.voc:
                vec[self.voc.index(word)] += 1
        return vec

    def get_utter_emb(self, text):
        embs = [self.w2v[word] for word in text.split(' ')\
            if word and word in self.w2v]
        if not embs:
            return np.zeros([self.w2v_dim],np.float32)
        return np.mean(embs,axis=0)

    def extract_entities(self, text):
        entities = {}
        for word in text.split(' '):
            for key,vals in self.entity_dict.iteritems():
                if word in vals:
                    entities[key] = word
        return entities

if __name__ == '__main__':
    get_fn = lambda fn:os.path.join('..','data',fn)
    nlu = NLU(get_fn('voc.txt'),get_fn('w2v.model'),300,{})
    text = "i'd like to book a table with italian food"
    print nlu.get_bow_vector(text)
    print nlu.get_utter_emb(text)
    print nlu.extract_entities(text)

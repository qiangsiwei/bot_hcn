# -*- coding: utf-8 -*-

import numpy as np
from utils import *
from nlu import NLU
from dst import DST
from models import HCN

class BotHCN(object):

    def __init__(self, voc_fn, w2v_fn, w2v_dim, entity_types, entity_dict,
            action_mask_dict, obs_size, act_size, templates):
        self.nlu = NLU(voc_fn,w2v_fn,w2v_dim,entity_dict)
        self.dst = DST(entity_types)
        self.model = HCN(action_mask_dict,obs_size,act_size)
        self.templates = templates

    def train(self, data_fn, epochs=5):
        def train_dialog(dialog):
            loss = 0
            self.dst.clear()
            self.model.reset_state()
            for text,action in dialog:
                feat_bow = self.nlu.get_bow_vector(text)
                feat_emb = self.nlu.get_utter_emb(text)
                entities = self.nlu.extract_entities(text)
                self.dst.update(entities)
                feat_ctx = self.dst.get_feat()
                feats = np.concatenate((feat_bow,feat_emb,feat_ctx),axis=0)
                action_mask = self.model.get_action_mask(feat_ctx)
                loss += self.model.train_step(feats,action,action_mask)[0]
            return loss
        data = list(get_data(data_fn))
        data = convert_train_data(data,self.templates)
        data_train = data[:int(.9*len(data))]
        data_valid = data[int(.9*len(data)):]
        for epoch in xrange(epochs):
            loss = sum([train_dialog(dialog) for dialog in data_train])
            accu = self.eval(data_valid)
            print '[{0}/{1}] {2:.4f} {3:.4f}'.format(epoch,epochs,loss,accu)
        self.model.save()

    def eval(self, dialogs):
        def eval_dialog(dialog):
            correct = 0
            self.dst.clear()
            self.model.reset_state()
            for text,real in dialog:
                feat_bow = self.nlu.get_bow_vector(text)
                feat_emb = self.nlu.get_utter_emb(text)
                entities = self.nlu.extract_entities(text)
                self.dst.update(entities)
                feat_ctx = self.dst.get_feat()
                feats = np.concatenate((feat_bow,feat_emb,feat_ctx),axis=0)
                action_mask = self.model.get_action_mask(feat_ctx)
                pred = self.model.predict_action(feats,action_mask)
                correct += int(pred==real)
            return 1.*correct/len(dialog)
        return 1.*sum([eval_dialog(dialog) for dialog in dialogs])/len(dialogs)

    def test(self):
        self.dst.clear()
        self.model.load()
        self.model.reset_state()
        while True:
            text = raw_input(':: ')
            if text in ('clear','reset','restart'):
                self.dst.clear()
                self.model.reset_state()
                print ''
            elif text in ('exit','quit','stop'):
                break
            else:
                text = text or '<SILENCE>'
                feat_bow = self.nlu.get_bow_vector(text)
                feat_emb = self.nlu.get_utter_emb(text)
                entities = self.nlu.extract_entities(text)
                self.dst.update(entities)
                feat_ctx = self.dst.get_feat()
                feats = np.concatenate((feat_bow,feat_emb,feat_ctx),axis=0)
                action_mask = self.model.get_action_mask(feat_ctx)
                pred = self.model.predict_action(feats,action_mask)
                print '>>', self.templates[pred].format(**self.dst.entities)

if __name__ == '__main__':
    pass

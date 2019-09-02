# -*- coding: utf-8 -*-

import numpy as np

class DST(object):

    def __init__(self, entity_types):
        self.entity_types = entity_types
        self.entities = {key:None for key in self.entity_types}
        self.num_features = len(self.entities)

    def clear(self):
        self.entities = {key:None for key in self.entity_types}

    def update(self, entities):
        for key,val in entities.iteritems():
            self.entities[key] = val

    def get_feat(self):
        return np.array([int(bool(self.entities[key]))\
            for key in self.entity_types])

if __name__ == '__main__':
    pass

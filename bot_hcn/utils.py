# -*- coding: utf-8 -*-

import os, re, operator

def get_data(fn):
    with open(fn) as fin:
        data = filter(lambda x:not re.match(r'^resto_',x[0]),
            map(lambda x:re.sub(r'^\d+\s','',x).split('\t'),
                fin.read().split('\n')))
    dialog = []
    for turn in data:
        if not turn[0]:
            yield dialog
            dialog = []
        else:
            dialog.append(turn)

def convert_train_data(data, templates):
    for i,dialog in enumerate(data):
        for j,turn in enumerate(dialog):
            turn[1] = re.sub(r'(?<=what do you think of this option:).+','',turn[1])
            turn[1] = re.sub(r'(?<=here it is resto_).+(?=phone|address)','',turn[1])
            turn[1] = re.sub(r'(?<=api_call ).+','{cuisine} {location} {party_size} {rest_type}',turn[1])
            data[i][j][1] = templates.index(turn[1])
    return data

if __name__ == '__main__':
    data = list(get_data(os.path.join('..','data','babi-task5.txt')))
    with open(os.path.join('..','data','voc.txt'),'w') as fout:
        fout.write('\n'.join(sorted(set(
            reduce(operator.add,map(lambda x:x[0].split(),
                reduce(operator.add,data,[])),[])),key=len)))
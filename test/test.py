# -*- coding: utf-8 -*-

import os, sys
sys.path.append('..')
from bot_hcn.bot_hcn import BotHCN

def train(bot, data_fn):
    bot.train(data_fn=data_fn)

def test(bot):
    bot.test()

if __name__ == '__main__':
    templates = [
        "any preference on a type of cuisine",
        "api_call {cuisine} {location} {party_size} {rest_type}",
        "great let me do the reservation",
        "hello what can i help you with today",
        "here it is resto_address",
        "here it is resto_phone",
        "how many people would be in your party",
        "i'm on it",
        "is there anything i can help you with",
        "ok let me look into some options for you",
        "sure is there anything else to update",
        "sure let me find an other option for you",
        "what do you think of this option:",
        "where should it be",
        "which price range are looking for",
        "you're welcome"]
    get_fn = lambda fn:os.path.join('..','data',fn)
    entity_types = ['cuisine','location','party_size','rest_type']
    entity_dict = {
        'cuisine'    :['british','cantonese','french','indian','italian','japanese','korean','spanish','thai','vietnamese'],
        'location'   :['bangkok','beijing','bombay','hanoi','paris','rome','london','madrid','seoul','tokyo'],
        'party_size' :['1','2','3','4','5','6','7','8','one','two','three','four','five','six','seven','eight'],
        'rest_type'  :['cheap','moderate','expensive']}
    action_mask_dict = {
        '0000' :[4,8,1,14,7,15],
        '0001' :[4,8,1,14,7],
        '0010' :[4,8,1,14,15],
        '0011' :[4,8,1,14],
        '0100' :[4,8,1,7,15],
        '0101' :[4,8,1,7],
        '0110' :[4,8,1,15],
        '0111' :[4,8,1],
        '1000' :[4,8,14,7,15],
        '1001' :[4,8,14,7],
        '1010' :[4,8,14,15],
        '1011' :[4,8,14],
        '1100' :[4,8,7,15],
        '1101' :[4,8,7],
        '1110' :[4,8,15],
        '1111' :[2,3,5,6,8,9,10,11,12,13,16]}
    bot = BotHCN(
        voc_fn=get_fn('voc.txt'),
        w2v_fn=get_fn('w2v.model'),
        w2v_dim=300,
        entity_types=entity_types,
        entity_dict=entity_dict,
        action_mask_dict=action_mask_dict,
        obs_size=85+300+4,
        act_size=16,
        templates=templates)
    data_fn = os.path.join('..','data','babi-task5.txt')
    # train(bot,data_fn=data_fn)
    test(bot)

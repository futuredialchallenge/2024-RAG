"""
Copyright 2024 Tsinghua University
Author: Yucheng Cai (cyc22@mails.tsinghua.edu.cn)
"""

import json
import os
import logging
import random
import re
import copy

import numpy as np
import torch
from transformers import BertTokenizer

from config import global_config as cfg
from collections import OrderedDict
import metrics

special_tokens=[ '[EOS_U]', '[EOS_Q]', '[EOS_A]', '[EOS_S]', '[QA]', 
'g', 'mb', 'm', '【流量提醒】', '移动官网', '验证', '查询用户已办理的业务', '查询流量信息',
'查询特定业务信息', '[地名]', '[组织名]', '[人名]', '[开始时间]', '[结束时间]', '[手机号]',
'[身份证号]', '[数字]']

all_api_intents = {
'':'不需要进行查询',
'查询用户已办理的业务':'查询用户当前办理的业务',
'查询特定业务信息':'查询某个业务的一些信息',
'查询流量信息':'查询和流量有关的信息，包括是否超出，剩余多少',
'查询本月月租':'查询当月的月租',
'[QA]':'为用户推荐一些合适的套餐，或者从问答手册找出合适的回复'
}
"""
api_label_dict = {'查询用户已办理的业务' : 0,
'查询特定业务信息': 1,
'查询流量信息' : 2,
'[QA]' : 3,
'' : 4
}
"""

api_label_dict = {'查询用户已办理的业务' : 0,
'查询特定业务信息': 1,
'查询其他信息' : 2,
'[QA]' : 3,
'' : 4
}

api_label_dict_new = {'查询用户已办理的业务' : 0,
'查询特定业务信息': 1,
'查询其他信息' : 2,
# '查询流量信息'
'[QA]' : 3,
'' : 4
}
label_api_dict = {api_label_dict_new[k]:k for k in api_label_dict_new}

# 验证：
# 目前有个流量要领取下吗？
# 目前有个限时优惠的流量
# 目前有个限时优惠的定向流量
# 亲，目前有个限时优惠的定向流量需要帮您领取吗？
# 身份证号码（温馨提示：身份证号码尾号为字母X，请用大写输入，如果没有直接输入数字） 
# 麻烦稍等，正在查询中
# 正在帮您查询，请稍等。亲，感谢您的耐心等待

def split_api_result(api_result):
    splitted_results = []
    if '，业务名称：' in api_result: # to add '。业务名称：'
        splitted = api_result.split('，业务名称：')
        if splitted[0] not in splitted_results:
            splitted_results.append(splitted[0])
        for piece in splitted[1:]:
            append_piece = '业务名称：'+piece
            if append_piece not in splitted_results:
                splitted_results.append(append_piece)
    elif '。业务名称：' in api_result: # to add '。业务名称：'
        splitted = api_result.split('。业务名称：')
        if splitted[0] not in splitted_results:
            splitted_results.append(splitted[0])
        for piece in splitted[1:]:
            append_piece = '业务名称：'+piece
            if append_piece not in splitted_results:
                splitted_results.append(append_piece)
    else:
        splitted_results.append(api_result)
    return splitted_results

# read data for training api classification
def read_data_api(tokenizer, ret_train=False):
    encoded_path = 'data/data_encoded_ret.json' if ret_train else 'data/data_encoded_api.json'
    exception_num = 0
    exception_turns = []
    if os.path.exists(encoded_path):
        encoded_data = json.loads(open(encoded_path, 'r').read())
    else:
        print('encoding data')
        #if ret_train:
        #data_path_train = 'data/train_processed_cleaned.json'
        #data_path_dev = 'data/dev_processed_cleaned.json'
        #data_path_test = 'data/test_processed_cleaned.json'
        data_path_train = 'data/train_final_processed.json'
        data_path_dev = 'data/dev_final_processed.json'
        data_path_test = 'data/test_final_processed.json'
        #else:
        #    data_path_train = 'data/train.json'
        #    data_path_dev = 'data/dev.json'
        #    data_path_test = 'data/test.json'
        train_data = json.loads(open(data_path_train, 'r').read())
        dev_data = json.loads(open(data_path_dev, 'r').read())
        # test_data = json.loads(open(data_path_test, 'r').read())
        source_data = {'train':train_data, 'dev':dev_data} # ,'test':test_data}

        encoded_data = {'train':[],'dev':[],'test':[]}
        count = 0
        #{"user": ","system": ,"api_query": "","api_result": ""}
        if ret_train:
            #kb_sv = json.loads(open('qa_cleaned_new.json', 'r').read())
            #kb_vs_des = json.loads(open('qa_cleaned_ret.json', 'r').read())
            _, kb_vs_des, _ = get_new_qa(return_result=True)
            kb_ret = json.loads(open('kb/ret_qa.json', 'r').read())
            #kb_vs = {kb_sv[s]:s for s in kb_sv}

            for key,data in source_data.items():
                for id, dial in data.items():
                    hist = ['']
                    local_kb = dial['local_kb']
                    for turn_num in range(len(dial['log'])):
                        turn = dial['log'][turn_num]
                        case = {}
                        case['id'] = id
                        case['turn_num'] = turn_num
                        case['local_kb'] = local_kb
                        cfg.retrieve_hist = 2
                        _ , hist = get_spoken(hist, [clean_utt(turn['user'].lower())], role='user')
                        tokenized = tokenizer(hist[0], max_length = 512, truncation=True) #, add_special_tokens=False)
                        api_query = clean_intent(turn['api_query'])
                        resp = clean_utt(turn['system'].lower())
                        #api_result, resp = delex(api_query, turn['api_result'], resp)
                        api_result, _ = delex(api_query, turn['api_result'], resp)
                        _ , hist = get_spoken(hist, [resp], role='system')
                        case['input'] = tokenized['input_ids']
                        case['attention_mask'] = tokenized['attention_mask']

                        if api_query in api_label_dict:
                            api_label = api_label_dict[api_query]
                        elif '查询用户已办理的业务' in api_query:
                            api_label = 0
                        elif api_query=='查询流量信息' or api_query=='查询其他信息':
                            api_label = 2
                        else:
                            api_label = api_label_dict['']
                            count += 1
                        if api_label==0 or api_label==2: # local_kb
                            if turn['api_result']!= '':
                                splitted_results = split_api_result(turn['api_result'])
                                l = []
                                for splitted_result in splitted_results:
                                    l.append(len(kb_ret) + local_kb.index(splitted_result)) # kb_ret.index(kb_vs[resp])
                                if l!=[]:
                                    case['labels'] = l
                                else:
                                    case['labels'] = [-1]
                                    exception_num += 1
                                # should consider multiple results
                        if api_label==1: # 查询特定业务信息
                            if turn['api_result']!= '':
                                splitted_results = split_api_result(turn['api_result'])
                                l = []
                                for splitted_result in splitted_results:
                                    if splitted_result in kb_ret:
                                        l.append(kb_ret.index(splitted_result))
                                        #case['labels'] = [kb_ret.index(kb_vs[resp])]
                                    else:
                                        #case['labels'] = [-1]
                                        exception_num += 1
                                #l.append(len(kb_ret) + local_kb.index(splitted_result)) # kb_ret.index(kb_vs[resp])
                                if l!=[]:
                                    case['labels'] = l
                                else:
                                    case['labels'] = [-1]
                        if api_label==3: # qa
                            if resp in kb_vs_des:
                                if kb_vs_des[resp] in kb_ret:
                                    case['labels'] = [kb_ret.index(kb_vs_des[resp])]
                                else:
                                    case['labels'] = [-1]
                                    exception_turns.append(resp)
                                    exception_num += 1
                            else:
                                case['labels'] = [-1]
                                exception_turns.append(resp)
                                if key != 'test':
                                    exception_num += 1

                        if api_label==4:
                            case['labels'] = [-1]
                        case['api_label'] = api_label
                        case['domain'] = api_label
                        encoded_data[key].append(case)
                        if 'labels' not in case:
                            case['labels'] = [-1]
        else:
            for key,data in source_data.items():
                for id, dial in data.items():
                    hist = ['']
                    for turn_num in range(len(dial)):
                        turn = dial[turn_num]
                        case = {}
                        case['id'] = id
                        case['turn_num'] = turn_num
                        cfg.retrieve_hist = 2
                        _ , hist = get_spoken(hist, [clean_utt(turn['user'].lower())], role='user')
                        tokenized = tokenizer(hist[0], max_length = 512, truncation=True) #, add_special_tokens=False)
                        api_query = clean_intent(turn['api_query'])
                        resp = clean_utt(turn['system'].lower())
                        api_result, resp = delex(api_query, turn['api_result'], resp)
                        _ , hist = get_spoken(hist, [resp], role='system')
                        case['input'] = tokenized['input_ids']
                        case['attention_mask'] = tokenized['attention_mask']

                        if api_query in api_label_dict:
                            api_label = api_label_dict[api_query]
                        elif '查询用户已办理的业务' in api_query:
                            api_label = 0
                        elif api_query=='查询流量信息' :
                            api_label = 2
                        else:
                            api_label = api_label_dict['']
                            count += 1
                        case['domain'] = api_label
                        encoded_data[key].append(case)
        print(count)
        print(exception_num)
        # sum(case['api_label']==3 for case in dev)
        json.dump(encoded_data, open(encoded_path, 'w'), indent=2)       
    return encoded_data

def clean_dataset_qa():
    files= ['data/dev.json', 'data/train.json', 'data/test.json']
    for file in files:
        file_new = file.replace('.json', '_new.json')
        data = json.load(open(file, 'r'))
        for id, dial in data.items():
            """
            turn = dial[0]
            if '人工' in turn['user'] and turn['api_query']=='[QA]':
                dial[0]['api_query'] = ''
            turn = dial[0]
            if '季节' in turn['system'] and turn['api_query']=='[QA]':
                dial[0]['api_query'] = ''
            turn1 = dial[-1]
            if '好评' in turn1['system'] and turn1['api_query']=='[QA]':
                dial[-1]['api_query'] = ''
            """
            for turn in dial:
                if '满意' in turn['system'] and turn['api_query']=='[QA]' and len(turn['system'])<150:
                    turn['api_query'] = ''
        json.dump(data, open(file_new, 'w'), indent=2, ensure_ascii=False)
    return

def clean_dataset():
    files= ['data/dev.json', 'data/train.json', 'data/test.json']
    for file in files:
        file_new = file.replace('.json', '_new.json')
        data = json.load(open(file, 'r'))
        for id, dial in data.items():
            for turn in dial:
                turn['user'] = clean_utt(turn['user'])
                turn['system'] = clean_utt(turn['system'])
        json.dump(data, open(file_new, 'w'), indent=2, ensure_ascii=False)
    return

def clean_qa():
    file = 'qa_cleaned.json'
    data = json.load(open(file, 'r'))
    new_data = {}
    for q,a in data.items():
        new_data[clean_utt(q)] = clean_utt(a)
    json.dump(new_data, open(file, 'w'), indent=2, ensure_ascii=False)
    return

def clean_utt(utt):
    toxics = ['(涉敏话术)', 'txt', '**$%', '^&**', '（*^▽^*）', '(*^▽^*)',
     'o(*￣-￣*)o', '\t', '？？', '? ? ?', '~', '（*^__^*）', '\\n', '*^__^*',
     '亲，请问还在吗？', '长时间不回复 系统会自动掉线哦', '（?￣ ? ￣?）', '“',
      '”', 'online service', '亲，请问还在线吗？', '---', '--', 'o（*￣-￣*）o',
      '（温馨提示：您可以通过稍后下发的短信链接查询问题解决进展哦）','亲，对不起让您久等了！',
      ' changjingzhi进人工入口'
      # （温馨提示：您可以通过稍后下发的短信链接查询问题解决进展哦）
      ]
    # 详情页
    utt_new = utt
    utt_new = utt_new.replace('。。。', '。').replace('。。', '。').replace('??', '好的').replace('！。', '！')
    utt_new = utt_new.replace('？。', '。')
    for toxic in toxics:
        utt_new = utt_new.replace(toxic, '')
    utt_new = re.sub(r'\[[^\[\]]*\]', '', utt_new) # clear [em_num]
    url = re.compile('http[^(\u4e00-\u9fa5)]+')
    utt_new = re.sub(url, '移动官网', utt_new)
    if 'yinyong' in utt:
        utt_new = re.sub(r'yinyong[^()]*yinyong', '', utt_new)
    if "详情页" in utt_new:
        utt_new = re.sub(r'详情页[^详情页查看]*查看', '详情页查看', utt_new)
    return utt_new

def clean_intent(intent):
    handle = ''
    # 方案描述=业务内容
    intent_new = intent
    if '查询当前套餐' in intent:
        if '查询用户已办理的业务' in intent:
            intent_new = intent_new.replace('查询当前套餐，', '').replace('查询当前套餐；', '').replace('查询当前套餐', '')
        else:
            intent_new = intent_new.replace('查询当前套餐', '查询用户已办理的业务')
    if '查询本月月租' in intent:
        if '查询用户已办理的业务' in intent:
            intent_new = intent_new.replace('查询本月月租，', '').replace('查询本月月租；', '').replace('查询本月月租', '')
        else:
            intent_new = intent_new.replace('查询本月月租', '查询用户已办理的业务')
    if '查询定向流量信息' in intent:
        if '查询特定业务信息' in intent:
            intent_new = intent_new.replace('查询定向流量信息，', '').replace('查询定向流量信息；', '').replace('查询特定业务信息', '')
        else:
            intent_new = intent_new.replace('查询定向流量信息', '查询特定业务信息')
    if '查询剩余流量情况' in intent or '查询超出流量' in intent or '查询流量信息' in intent:
        intent_new = intent_new.replace('查询超出流量', '查询其他信息').replace('查询剩余流量情况', '查询其他信息').replace('查询流量信息', '查询其他信息')
    """
    if '取消' in intent:
        if '验证' in intent:
            intent_new = intent_new.replace('取消，', '').replace('取消；', '').replace('取消', '')
        else:
            intent_new = intent_new.replace('取消', handle)
    if '办理' in intent and ('查询用户已办理的业务' not in intent):
        if '验证' in intent:
            intent_new = intent_new.replace('办理，', '').replace('办理；', '').replace('办理', '')
        else:
            intent_new = intent_new.replace('办理', handle)
    intent_new = intent_new.replace('验证', handle)   
    intent_new = intent_new.replace('推荐', '')   
    """  
    return intent_new

def delex(query, result, resp):
    # double mark that quote turns before
    # （涉敏话术）
    # 月租 no need for delex
    if (('查询超出流量' in query) or ('查询流量信息' in query)) and "【流量提醒】" in resp:
        result_delex = re.sub(r'【流量提醒】[^【】]*【中国移动】', '【流量提醒】', result)
        resp_delex = re.sub(r'【流量提醒】[^【】]*【中国移动】', '【流量提醒】', resp)
    else:
        result_delex = result
        resp_delex = resp
    return result_delex, resp_delex

def convert_to_sequences(data, posterior=False, return_dict=False):
    sequences=[]
    dicts=[]
    special_turns =[]
    KB_count = 0
    for id, dial in data.items():
        new_dial = {"log":[]}
        for turn in dial['log']:
            turn_dict = {}
            hist = ['']
            if return_dict:
                _ , hist = get_spoken(hist, [clean_utt(turn['user'].lower())], role='user')

                turn_dict['user'] = clean_utt(turn['user'].lower())+'[EOS_U]'
                api_query = clean_intent(turn['api_query'])
                resp = clean_utt(turn['system'].lower())
                api_result, resp = delex(api_query, turn['api_result'], resp)
                if api_query!='[QA]' :
                    turn_dict['api_result'] = '参考知识：' + api_result.lower() + '[EOS_A]'
                else:
                    qa_result = f"问题：{hist[0]}, 答案：{resp}"
                    turn_dict['api_result'] = '参考知识：' + qa_result + '[EOS_A]'
                turn_dict['api_query'] = api_query + '[EOS_Q]'
                #if api_query=='[QA]' :
                #    turn_dict['resp'] = '[EOS_S]'
                #if api_query=='[QA]' and len(resp)>100:
                #    turn_dict['resp'] = resp[:50].lower() + '[EOS_S]'
                #else:
                turn_dict['resp'] = resp.lower() + '[EOS_S]'
                turn_dict['local_kb'] = dial['local_kb']
                new_dial['log'].append(turn_dict)
                _ , hist = get_spoken(hist, [resp], role='system')

        if return_dict:
            dicts.append(new_dial)
    if return_dict:
        #kb_bad_path = 'data/extra/bad_kb.json'
        #json.dump(special_turns, open(kb_bad_path, 'w'),indent=2,ensure_ascii=False)
        return dicts
    else:
        return sequences

def serialize_kb(KB):
    kb_seq = []
    for e in KB:
        if e == 'NA':
            NA_temp = []
            for prop in KB['NA']:
                NA_temp.append(prop+':'+KB['NA'][prop])
            kb_seq.append(';'.join(NA_temp))
        else:
            ent_info = KB[e]
            ent_temp = []
            for ent in ent_info:
                if ent == 'name':
                    ent_temp.append('名称:'+ent_info[ent])
                elif ent == 'type':
                    ent_temp.append('类型:'+ent_info[ent])
                else:
                    ent_temp.append(ent+':'+ent_info[ent])
            kb_seq.append(';'.join(ent_temp))
    kb_seq = ';'.join(kb_seq)
    return kb_seq

def get_dial_hist(hist,turn):
    max_len = cfg.retrieve_hist
    new_hist = []
    hist_seq = ''
    if len(hist)>=max_len:
        for i in range(len(hist)-1):
            temp = hist[i]
            new_hist.append(hist[i+1])
            hist_seq = ' 用户：' + temp ['用户'] + ' 客服：' + temp['客服']
        new_hist.append(turn)
        hist_seq = hist_seq + ' 用户：' + turn['用户']
    else:
        for temp in hist:
            hist_seq = ' 用户：' + temp ['用户'] + ' 客服：' + temp['客服']
        new_hist = hist
        new_hist.append(turn)
        hist_seq = hist_seq + ' 用户：' + turn['用户']
    return new_hist, hist_seq

def get_pseudo_retrieval_data(tokenizer):
    # gpt_tokenizer is loaded for decoding previous pseudo label
    gpt_tokenizer = BertTokenizer.from_pretrained('experiments/baseline_post_gtdb/best_post_model')

    pseudo_data = json.load(open(cfg.pseudo_path,'r'))
    new_data = []
    special = []
    for dial in pseudo_data:
        new_dial = {'KB':[], 'content':[]}
        for turn in dial:
            new_turn = {}
            new_turn['用户'] = gpt_tokenizer.decode(turn['user'][1:-1]).replace(' ','')
            new_turn['客服'] = gpt_tokenizer.decode(turn['resp'][:-1]).replace(' ','')

            turn_kb = gpt_tokenizer.decode(turn['db_gt'][:-1]).replace(' ','').split('；')    
            turn_kb = list(set(turn_kb)) # remove repetition
            for item in turn_kb:
                flag = 0
                if ':' not in item:
                    turn_kb.remove(item)
                else:
                    values = item.split(':')[-1]
                    value = values.split(',') if (',' in values) else [values]
                    for v in value:
                        if v in new_turn['客服']:
                            flag = 1
                    if flag == 0:
                        special.append((new_turn['客服'], item))
                        turn_kb.remove(item)
            new_turn['kb'] = turn_kb
            new_dial['content'].append(new_turn)
            new_dial['KB'].extend(turn_kb)
        new_data.append(new_dial)
    cases=[]
    for dial in new_data:
        hist = []
        KB=dial['KB']
        for turn in dial['content']:
            positive = [] # kv_pairs
            negative = [] # kv_pairs
           
            # construct positive and negative sample from KB
            for case in turn['kb']:
                positive.append(tokenizer.encode(case))
            for item in KB:
                flag = 0
                values = item.split(':')[-1]
                value = values.split(',') if (',' in values) else [values]
                for v in value:
                    if v in turn['客服']:
                        flag = 1
                if flag == 0:
                    negative.append(tokenizer.encode(item))
            hist, hist_seq = get_dial_hist(hist, turn)
            seq = tokenizer.encode(hist_seq)
            for p in positive:
                cases.append({'context':seq,'triple':p,'label':1})
            random.shuffle(negative)
            SAMPLENUM = int(len(negative)*0.3)
            for n in negative[:SAMPLENUM]:
                cases.append({'context':seq,'triple':n,'label':-1})
    print(len(positive), len(negative))
    return cases

def get_retrieval_data(data, tokenizer, dial_ids=None, ebm=False):
    cases=[]
    max_len = 5
    for dial in data:
        #new_dial = []
        hist = []
        dial_id=dial['id']
        if dial_ids is not None and dial_id not in dial_ids:
            continue
        KB=dial['KB']
        spoken = ''
        for turn in dial['content']:
            turn_cases = []
            # note that spoken also includes the content in this turn
            spoken = spoken + turn['用户'] + turn['客服']
            positive = [] # kv_pairs
            negative = [] # kv_pairs
            #if 'info' in turn:
            #    for ent in turn['info']['ents']:
            #    for triple in turn['info']['triples']: 
            # construct positive and negative sample from KB
            for _,kvs in KB.items():
                for k,v in  kvs.items():
                    if k !='type': # remove type for now
                        if (',' not in v) and (v!= ''):
                            if v not in spoken:
                                negative.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ v))
                            elif v in turn['客服']:
                                positive.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ v))
                        else:
                            tmp_v = []
                            flag = 0
                            if cfg.fine_grained:
                                for value in v.split(','):
                                    if value not in spoken:
                                        negative.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ value))
                                    elif value in turn['客服']:
                                        positive.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ value))
                            else:
                                for value in v.split(','):
                                    if value not in spoken:
                                        tmp_v.append(value)
                                    elif value in turn['客服']:
                                        tmp_v.append(value)
                                        flag = 1
                                if (flag == 0) and (tmp_v!=[]):
                                    negative.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ ','.join(tmp_v)))   
                                elif flag == 1:
                                    positive.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ ','.join(tmp_v)))
            hist, hist_seq = get_dial_hist(hist, turn)
            seq = tokenizer.encode(hist_seq)
            #cases.append({'context':tokenizer.encode(hist_seq), 'positive':positive, 'negative':negative}) # change case to hist, triple, label
            for p in positive:
                #if cfg.only_one_model:
                #    cases.append({'context':hist_seq,'triple':p,'label':1})
                #else:
                turn_cases.append({'context':seq,'triple':p,'label':1})
            for n in negative:
                turn_cases.append({'context':seq,'triple':n,'label':-1})
            if not ebm:
                cases.extend(turn_cases)
            else:
                if turn_cases!=[]:
                    turn_case = {'context':hist_seq, 'cases':turn_cases}
                    cases.append(turn_case)

    return cases

def get_retrieval_sequence(tokenizer, seqs, triples):
    final = [(seqs[s] + '[SEP]' + triples[s]).replace('[UNK]','g') for s in range(len(seqs))]
    tokenized = tokenizer(final, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return tokenized

def read_data(tokenizer, posterior = False, return_dict =False, retrieve=False, ebm=False):
    if posterior:
        encoded_path=os.path.join(cfg.data_dir, 'encoded_data_post.json')
    else:
        encoded_path=os.path.join(cfg.data_dir, 'encoded_data.json')
    if return_dict:
        encoded_path=os.path.join(cfg.data_dir, 'encoded_data_dict.json')
    if retrieve:
        if cfg.fine_grained:
            encoded_path=os.path.join(cfg.data_dir, 'encoded_data_retrieve_fg.json')
        else:
            encoded_path=os.path.join(cfg.data_dir, f"encoded_data_retrieve_{cfg.retrieve_hist}_{cfg.pseudo_label_retrieval}.json")
    if not os.path.exists(encoded_path):
        dev_path = 'data/dev_final_processed.json'
        test_path = 'data/test_final_processed.json'
        train_data = json.load(open(cfg.data_path, 'r', encoding='utf-8')) # 'data/train_final_processed.json'
        dev_data = json.load(open(dev_path, 'r', encoding='utf-8'))
        # test_data = json.load(open(test_path, 'r', encoding='utf-8'))
        # get seq data
        logging.info('Encoding data ...')
        if retrieve:
            encoded_data = {}
            encoded_data['train'] = []
            #if cfg.pseudo_label_retrieval:
            #    encoded_data['train'].extend(get_pseudo_retrieval_data(tokenizer))
            encoded_data['train'].extend(get_retrieval_data(train_data, tokenizer))
            encoded_data['dev'] = get_retrieval_data(dev_data, tokenizer)
            # encoded_data['test'] = get_retrieval_data(test_data, tokenizer)
        elif ebm:
            encoded_data = {}
            encoded_data['train'] = []
            #if cfg.pseudo_label_retrieval:
            #    encoded_data['train'].extend(get_pseudo_retrieval_data(tokenizer))
            encoded_data['train'].extend(get_retrieval_data(train_data, tokenizer, ebm=True))
            encoded_data['dev'] = get_retrieval_data(dev_data, tokenizer, ebm=True)
            encoded_data['test'] = get_retrieval_data(test_data, tokenizer, ebm=True)
        else:
            train_seqs=convert_to_sequences(train_data, posterior, return_dict = return_dict)
            dev_seqs=convert_to_sequences(dev_data, posterior, return_dict = return_dict)
            test_seqs=convert_to_sequences(test_data, posterior, return_dict = return_dict)
            logging.info('Dialogs -- Train:{}, dev:{}, test:{}'.format(len(train_data), len(dev_data), len(test_data)))
            logging.info('Sequences -- Train:{}, dev:{}, test:{}'.format(len(train_seqs), len(dev_seqs), len(test_seqs)))
            seq_data={
                'train':train_seqs,
                'dev':dev_seqs
            }
            # , 'test':test_seqs }
            if not return_dict:
                json.dump(seq_data, open(os.path.join(cfg.data_dir, 'all_data.json'), 'w'),indent=2, ensure_ascii=False)
            else:
                encoded_data={}
                for s in ['train', 'dev']: # , 'test']:
                    encoded_data[s]=[]
                    for seq in seq_data[s]:
                        encoded_dial = []
                        for turn in seq['log']:
                            encoded_turn = {}
                            for key,value in turn.items():
                                if key!= 'local_kb':
                                    encoded_turn[key] = tokenizer.encode(value)[1:-1]
                                else:
                                    encoded_turn[key] = value
                            encoded_dial.append(encoded_turn)
                        encoded_data[s].append(encoded_dial)
        json.dump(encoded_data, open(encoded_path, 'w'))
        logging.info('Data encoded, saved in:{}'.format(encoded_path))
    else:
        logging.info('Reading encoded data from:{}'.format(encoded_path))
        encoded_data=json.load(open(encoded_path, 'r'))
        """
        if cfg.with_pseudo_label :
            pseudo_data = json.load(open(cfg.pseudo_path, 'r'))
            random.shuffle(pseudo_data)
            pseudo_num = cfg.pseudo_porportion*len(encoded_data['train'])
            encoded_data['train'].extend(pseudo_data[:pseudo_num])
        """
    logging.info('Train:{}, dev:{}, test:{}'.format(len(encoded_data['train']), len(encoded_data['dev']))) # , len(encoded_data['test'])
    return encoded_data

def convert_to_dict(tokenizer,data):
    new_data = []
    for dial in data:
        new_dial=[]
        for turn in dial:
            enc = {}
            if '[SPEAKER 2]' in turn:
                enc['user'] = tokenizer.encode('[EOS_L]' + turn['[SPEAKER 2]'].lower()+'[EOS_U]')[1:-1]
            elif '[SPEAKER <B>]' in turn:
                enc['user'] = tokenizer.encode('[EOS_L]' + turn['[SPEAKER <B>]'].lower()+'[EOS_U]')[1:-1]
            else:
                break
            if '[SPEAKER 1]' in turn:
                enc['resp'] = tokenizer.encode(turn['[SPEAKER 1]'].lower()+'[EOS_S]')[1:-1]
            elif '[SPEAKER <A>]' in turn:
                enc['resp'] = tokenizer.encode(turn['[SPEAKER <A>]'].lower()+'[EOS_S]')[1:-1]
            else:
                break
            if enc!={}:
                new_dial.append(enc)
        if new_dial!=[]:
            new_data.append(new_dial)
    return new_data

def get_unsup(tokenizer, dial_num = 0, pretrain = False):#,ratio
    unlab_num = cfg.jsa_unsup*dial_num  #controls the num of unlab dials, # 1, 2, 4, 9
    encoded_path=os.path.join(cfg.data_dir, 'encoded_data_unl_'+str(unlab_num)+'.json')
    if pretrain:
        encoded_path=os.path.join(cfg.data_dir, 'encoded_data_unl_whole.json') if cfg.gpt else os.path.join(cfg.data_dir, 'encoded_data_unl_whole_t5.json')
    if not os.path.exists(encoded_path):
        unl=json.load(open('data/data_unlabel_process.json', 'r', encoding='utf-8'))
        #total_num = turn_num * ratio//10   #average dialog length, can be adjusted
        random.shuffle(unl)
        if pretrain:
            data = unl
        else:
            data = unl[:unlab_num] #data with low ratio are contained in data with high ratio
        logging.info('Encoding data ...')
        encoded_data=convert_to_dict(tokenizer,data)        
        json.dump(encoded_data, open(encoded_path, 'w'))
        logging.info('Data encoded, saved in:{}'.format(encoded_path))
    else:
        logging.info('Reading encoded data from:{}'.format(encoded_path))
        encoded_data=json.load(open(encoded_path, 'r'))
    if pretrain:
        new_data = {}
        sequences = []
        for dial in encoded_data:
            for turn in dial:
                if cfg.gpt:
                    sequences.append([tokenizer.convert_tokens_to_ids('[CLS]')] + turn['user']+
                    turn['resp'] + [tokenizer.convert_tokens_to_ids('[SEP]')])#+ tokenizer.convert_tokens_to_ids(['[EOS_E]','[EOS_UI]','[EOS_K]','[EOS_SI]'])
                else:
                    sequences.append({'input':([tokenizer.convert_tokens_to_ids('[CLS]')] + turn['user']),
                    'output':(turn['resp'] + [tokenizer.convert_tokens_to_ids('[SEP]')])})
        piece=len(sequences)//10
        new_data['dev'] = sequences[9*piece:]
        new_data['train'] =sequences[:9*piece]
        return new_data
    else:
        return encoded_data

def extract_test_dial(data='test'):
    dial_ids=json.load(open(os.path.join(cfg.data_dir, 'dial_ids.json'), 'r', encoding='utf-8'))
    all_data=json.load(open(cfg.data_path, 'r', encoding='utf-8'))
    test_data=[]
    for dial in all_data:
        if dial['id'] in dial_ids[data]:
            test_data.append(dial)
    return test_data

def _bucket_by_turn(encoded_data):
    turn_bucket = {}
    for dial in encoded_data:
        turn_len = len(dial)
        #修改记录：对于turn_len=0情况的处理
        if turn_len==0:
            continue
        if turn_len not in turn_bucket:
            turn_bucket[turn_len] = []
        turn_bucket[turn_len].append(dial)
    del_l = []
    for k in turn_bucket:
        if k >= 5:
            del_l.append(k)
        logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
   
    return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

def _construct_mini_batch(data,batch_size):
    all_batches = []
    batch = []
    for dial in data:
        batch.append(dial)
        if len(batch) == batch_size:
            # print('batch size: %d, batch num +1'%(len(batch)))
            all_batches.append(batch)
            batch = []

    if len(batch)>0:
        all_batches.append(batch)
    return all_batches

def inverse_transpose_batch(turn_batch_list):
    """
    :param turn_batch_list: list of transpose dial batch
    """
    dialogs = []
    total_turn_num = len(turn_batch_list)
    # initialize
    for idx_in_batch, _ in enumerate(turn_batch_list[0]['user']):
        dialog = []
        for turn_n in range(total_turn_num):
            dial_turn = {}
            turn_batch = turn_batch_list[turn_n]
            for key, v_list in turn_batch.items():
                value = v_list[idx_in_batch]
                dial_turn[key] = value
            dialog.append(dial_turn)
        dialogs.append(dialog)
    return dialogs

def inverse_transpose_turn(turn_list):
    turn_num = len(turn_list)
    dialog = []
    for turn_idx in range(turn_num):
        dial_turn = {}
        turn = turn_list[turn_idx]
        for key, value in turn.items():
            if key=='dial_id':
                continue
            dial_turn[key] = value
        dialog.append(dial_turn)
    return dialog
    
def get_batches(dials,batch_size):
    # organize data by batches  
    
    turn_bucket = _bucket_by_turn(dials)
    all_batches = []
    num_training_steps = 0
    num_turns = 0
    num_dials = 0

    for k in turn_bucket:
        batches = _construct_mini_batch(turn_bucket[k],batch_size)
        if len(batches)==0:
            continue
        
        num_training_steps += k * len(batches)
        num_turns += k * len(turn_bucket[k])
        num_dials += len(turn_bucket[k])
        all_batches += batches
    #log_str += 'total batch num: %d\n' % len(all_batches)    
    random.shuffle(all_batches)
    return all_batches

def split_turn_batch(turn_batch, batch_size, other_batch=None):
    batches=[]
    other_batches=[]
    B=len(turn_batch['user'])
    for i in range(0, B, batch_size):
        new_turn_batch={}
        if other_batch:
            other_batches.append(other_batch[i:i+batch_size])
        for key in turn_batch:
            new_turn_batch[key]=turn_batch[key][i:i+batch_size]
        batches.append(new_turn_batch)
    if other_batch:
        return batches, other_batches
    else:
        return batches, None

def transpose_batch(batch):
    dial_batch = []
    turn_num = len(batch[0]) 
    for turn in range(turn_num):
        turn_l = {}
        for dial in batch:
            this_turn = dial[turn]
            for k in this_turn:
                if k not in turn_l:
                    turn_l[k] = []
                turn_l[k].append(this_turn[k])
        dial_batch.append(turn_l)
    return dial_batch

def convert_batch_t5(cls, sep, turn_batch, pv_batch, first_turn=False, posterior=False):
    inputs = {}
    labels = {}
    contexts = []
    label_contexts = []
    if first_turn:   
        if cfg.gt_db:
            batch_zipped=zip(turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], (turn_batch['db_gtfg'] if cfg.fine_grained else turn_batch['db_gt']),turn_batch['aspn'], turn_batch['resp'])
        else:
            batch_zipped=zip(turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], turn_batch['db'],turn_batch['aspn'], turn_batch['resp'])
        if cfg.db_change:
            batch_zipped = zip(turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], turn_batch['db_all'], turn_batch['aspn'], turn_batch['resp'])    
        for u, b, ent, db, a, r in batch_zipped:
            if posterior:
                if cfg.no_user_intent:
                    context = [cls] + u + r # + [sep]
                    label_context = db + a + [sep] # [cls] +
                else:
                    context=[cls] + u + r # + [sep]#add [cls] and [sep] token
                    label_context= ent + b + db + a + [sep] 
            else:
                if cfg.no_user_intent:
                    context = ([cls] + db + u) if cfg.kb_grounding else ([cls] + u + db)# + [sep]
                    label_context = a + r + [sep] # [cls] +
                else: # not support as multi imput and multi output needed
                    context = [cls] + u + ent + b + db + a + r + [sep]
                    label_context=(len(u)+1)*[cfg.pad_id] + ent + b + len(db)*[cfg.pad_id] + a + r + [sep]  # db can be pad in priori model
            contexts.append(context)
            label_contexts.append(label_context)
    else:
        if cfg.gt_db:
            batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], (turn_batch['db_gtfg'] if cfg.fine_grained else turn_batch['db_gt']), turn_batch['aspn'], turn_batch['resp'])
        else:
            batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], turn_batch['db'], turn_batch['aspn'], turn_batch['resp'])
        #if cfg.db_change:
        #    batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn'], 
        #        turn_batch['entity'], turn_batch['db_all'], turn_batch['aspn'], turn_batch['resp'])
        for ur, u, b, ent, db, a, r in batch_zipped:
            if posterior: # act_change not added
                if cfg.no_user_intent:
                    context = [cls] + u + r # + [sep] # [cls] + ur + u + r + db + a + [sep]
                    label_context = db + a + [sep] # [cls] +
                else:
                    context = [cls] + ur + u + r # + [sep]
                    label_context= ent + b + db + a + [sep]
            elif cfg.db_change:
                context = [cls] + db + ur  + u # + [sep]
                label_context = b + a + r + [sep]
            else:
                if cfg.no_user_intent:
                    context = ([cls] + db + u) if cfg.kb_grounding else ([cls] + u + db)
                    # + [sep] # [cls] + ur + u + db + a + r + [sep]
                    label_context = a + r + [sep] # ur [cls] +
                else: # not supported
                    context = [cls] + ur + u + ent + b + db + a + r + [sep]
                    label_context=(len(ur+u)+1)*[cfg.pad_id] + ent + b + len(db)*[cfg.pad_id] + a + r + [sep]
            contexts.append(context)
            label_contexts.append(label_context)
    inputs['contexts'] = contexts
    inputs['contexts_np'], inputs['attention'] = padSeqs_gpt(inputs['contexts'], cfg.pad_id, attention=True)
    labels['contexts'] = label_contexts
    labels['contexts_np'], labels['attention'] = padSeqs_gpt(labels['contexts'], cfg.pad_id, attention=True)
    return inputs, labels
    
def convert_batch_turn(config, cls, sep, turn_batch, pv_batch, first_turn=False, posterior=False, qa=None):

    inputs = {}
    labels = {}
    contexts = []
    label_contexts = []
    if first_turn:   
        batch_zipped=zip(turn_batch['user'], turn_batch['api_query'], 
            turn_batch['api_result'], turn_batch['resp'])
        """
        elif cfg.joint_training:
            batch_zipped=zip(turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], turn_batch['db_retrieval'],turn_batch['aspn'], turn_batch['resp'])
        else:
            batch_zipped=zip(turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], turn_batch['db'],turn_batch['aspn'], turn_batch['resp'])
        """   
        for u, q, a, r in batch_zipped:
            if posterior:
                context=[cls] + u + r + q + a + [sep]#add [cls] and [sep] token
                label_context=(len(u+r)+1)*[cfg.pad_id] + q + a + [sep] 
            else:
                if config.rag_training:
                    context = [cls] + u + a + [sep]
                    label_context=(len(u)+1)*[cfg.pad_id] + a + [sep]
                if config.only_response:
                    context = [cls] + u + ( ([qa] + a + [r[-1]]) if qa in q else (a + r)) + [sep]
                    label_context=(len(u)+1)*[cfg.pad_id] + (((len(a)+1)*[cfg.pad_id] + [r[-1]]) if qa in q else (len(a)*[cfg.pad_id] + r)) + [sep]
                elif config.no_retrieval:
                    context = [cls] + u +  r + [sep]
                    label_context=(len(u)+1)*[cfg.pad_id]  +  r + [sep]
                else:
                    context = [cls] + u + q + a + ([r[-1]] if qa in q else r) + [sep]
                    label_context=(len(u)+1)*[cfg.pad_id] + q + len(a)*[cfg.pad_id] + ([r[-1]] if qa in q else r) + [sep]  # db can be pad in priori model
            contexts.append(context)
            label_contexts.append(label_context)
    else:
        batch_zipped=zip(pv_batch, turn_batch['user'], turn_batch['api_query'], 
            turn_batch['api_result'], turn_batch['resp'])
        """
        elif cfg.joint_training:
            batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], (turn_batch['db_gtfg'] if cfg.fine_grained else turn_batch['db_retrieval']), turn_batch['aspn'], turn_batch['resp'])
        else:
            batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn'], 
                turn_batch['entity'], turn_batch['db'], turn_batch['aspn'], turn_batch['resp'])
        """
        for ur, u, q, a, r in batch_zipped:
            if posterior: # act_change not added
                context = [cls] + ur + u +r + q + a + [sep]
                label_context=(len(ur+ u + r)+1)*[cfg.pad_id] + q + a + [sep]#len(r)*[cfg.pad_id]
            else:
                if config.rag_training:
                    context = [cls] + ur + u + a + r + [sep]
                    label_context=(len(ur + u + a) + 1)*[cfg.pad_id] + r + [sep]
                elif config.only_response:
                    context = [cls] + ur + u  + ( ([qa] + a + [r[-1]]) if qa in q else (a + r)) + [sep]
                    label_context=(len(ur+u)+1)*[cfg.pad_id] + (((len(a)+1)*[cfg.pad_id] + [r[-1]]) if qa in q else (len(a)*[cfg.pad_id] + r))+ [sep]
                elif config.no_retrieval:
                    context = [cls] + ur + u +  r + [sep]
                    label_context = (len(ur + u) + 1)*[cfg.pad_id]  +  r + [sep]
                else:
                    context = [cls] + ur + u + q + a + ([r[-1]] if qa in q else r) + [sep]
                    label_context = (len(ur+u)+1)*[cfg.pad_id] + q + len(a)*[cfg.pad_id] + ([r[-1]] if qa in q else r) + [sep]
            contexts.append(context)
            label_contexts.append(label_context)
    inputs['contexts'] = contexts
    inputs['contexts_np'], inputs['lengths'] = padSeqs_gpt(inputs['contexts'], cfg.pad_id)
    labels['contexts']=label_contexts
    labels['contexts_np'], labels['lengths']= padSeqs_gpt(labels['contexts'], cfg.pad_id)
    return inputs, labels

def convert_eval_batch_turn(config, cls, turn_batch, pv_batch, mode='gen_query', q_gen=None, a_gen = None, posterior=False):
    eval_batch=[]
    assert mode in ['gen_query', 'gen_answer', 'gen_resp']
    if pv_batch is None:
        if mode=='gen_query':
            for u, r in zip(turn_batch['user'], turn_batch['resp']):
                context= ([cls] +  u + r) if posterior  else [cls] + u 
                eval_batch.append(context)
        if mode=='gen_answer':
            for u, r, q in zip(turn_batch['user'], turn_batch['resp'], q_gen):
                context= ([cls] + u + r + q) if posterior else [cls] + u + q
                eval_batch.append(context)
        if mode=='gen_resp':
            for u, r, q, a in zip(turn_batch['user'], turn_batch['resp'],q_gen, a_gen):
                if config.only_response:
                    context= ([cls] + u + r + a) if posterior else [cls] + u + a
                elif config.no_retrieval: 
                    context= ([cls] + u + r ) if posterior else [cls] + u
                else:
                    context= ([cls] + u + r + q + a) if posterior else [cls] + u + q + a
                eval_batch.append(context)
    else:
        if mode=='gen_query':
            for hist, u, r in zip(pv_batch, turn_batch['user'], turn_batch['resp']):
                context = ([cls] + hist + u + r) if posterior else [cls] + hist + u 
                eval_batch.append(context)
        if mode=='gen_answer':
            for hist, u, r, q in zip(pv_batch, turn_batch['user'], turn_batch['resp'], q_gen):
                context = ([cls] + hist +  u + r + q) if posterior else [cls] + hist + u + q # check hist
                eval_batch.append(context)
        if mode=='gen_resp':
            for hist, u, r, q, a in zip(pv_batch, turn_batch['user'], turn_batch['resp'], q_gen, a_gen):
                if config.only_response:
                    context= ([cls] + hist + u + r + a) if posterior else [cls] + hist + u + a
                elif config.no_retrieval: 
                    context= ([cls] + hist + u + r ) if posterior else [cls] + hist + u
                else:
                    context= ([cls] + hist + u + r + q + a) if posterior else [cls] + hist + u + q + a
                eval_batch.append(context)
    return eval_batch

def convert_eval_batch_t5(cls, turn_batch, pv_batch, mode='gen_bspn', bspn_gen=None, ent_gen=None, db_gen = None, a_gen = None, posterior=False):
    eval_batch=[]
    assert mode in ['gen_kb','gen_ar','gen_resp']
    if pv_batch is None:
        if mode=='gen_kb':
            #if cfg.act_change:
            for u, r, ent, b in zip(turn_batch['user'], turn_batch['resp'],ent_gen,bspn_gen):
                if cfg.no_user_intent:
                    context= [cls] + u + r if posterior else [cls] + u
                else:
                    context= [cls] + u + r + ent + b if posterior else [cls] + u + ent + b
                if (posterior and cfg.posterior_change):
                    context= [cls] + u + ent + b + r
                eval_batch.append(context)
        if mode=='gen_ar':
            if cfg.db_change:
                for u, r, b,db in zip(turn_batch['user'], turn_batch['resp'], bspn_gen, db_gen):
                    context= [cls] + db + u + r +  b if posterior else [cls] + db + u + b
                    eval_batch.append(context)
            else:
                for u, r, ent, b,db in zip(turn_batch['user'], turn_batch['resp'], ent_gen, bspn_gen, db_gen):
                    if cfg.no_user_intent:
                        context= [cls] + u + r + db if posterior else [cls] + u +db
                    else:
                        context= [cls] + u + r + ent + b + db if posterior else [cls] + u + ent + b +db
                    if (posterior and cfg.posterior_change):
                        context= [cls] + u + ent + b + r + db
                    eval_batch.append(context)
    else:
        if mode=='gen_kb':
            for hist, u, r, ent, b in zip(pv_batch, turn_batch['user'], turn_batch['resp'],ent_gen,bspn_gen):
                if cfg.no_user_intent:
                    context = [cls] + u + r if posterior else [cls] + u # hist + 
                else:
                    context = [cls] + hist + u +r  + ent + b if posterior else [cls] + hist + u + ent + b
                if (posterior and cfg.posterior_change):
                    context= [cls] + hist + u + ent + b + r
                eval_batch.append(context)
        if mode=='gen_ar':
            if cfg.db_change:
                for hist,u, r, b,db in zip(pv_batch, turn_batch['user'], turn_batch['resp'], bspn_gen, db_gen):
                    context= [cls] + db + hist + u + r +  b if posterior else [cls] + db + hist + u + b
                    eval_batch.append(context)
            else:
                for hist, u, r, ent, b, db in zip(pv_batch, turn_batch['user'], turn_batch['resp'],ent_gen,bspn_gen, db_gen):
                    if cfg.no_user_intent:
                        context= [cls] + u + r + db if posterior else [cls] + u + db # hist+
                    else:
                        context = [cls] + hist+ u + r + ent + b + db if posterior else [cls] + hist+ u + ent + b + db
                    if (posterior and cfg.posterior_change):
                        context= [cls] + hist + u + ent + b + r + db
                    eval_batch.append(context)
    return eval_batch

def padSeqs_gpt(sequences, pad_id, maxlen=None, attention=False):
    lengths = []
    for x in sequences:
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_mexlen = np.max(lengths)
    # maxlen = 512
    if seq_mexlen > 512: 
        maxlen = 512
    else:
        maxlen = seq_mexlen
    
    x = (np.ones((num_samples, maxlen)) * pad_id)
    attentions = (np.ones((num_samples, maxlen)) * pad_id)
    for idx, s in enumerate(sequences):
        if not len(s):
            print('empty list was found in padSeqs')
        trunc = s[-maxlen:]
        trunc = np.asarray(trunc)
        x[idx, :len(trunc)] = trunc
        attentions[idx, :len(trunc)] = np.ones(len(trunc))
    if attention:
        return x, attentions
    else:  
        return x, lengths

def train_collate_fn(batch):
    if isinstance(batch[0],dict):
        batch_tensor = {}
        for key in batch[0]:
            input = [b[key] for b in batch]
            padded = padSeqs(input, cfg.pad_id)
            batch_tensor[key] = torch.from_numpy(padded).long()
    else:
        pad_batch = padSeqs(batch, cfg.pad_id)
        batch_tensor=torch.from_numpy(pad_batch).long()
    return batch_tensor

def test_collate_fn(batch, sep_id):
    # prediction
    # sep_id: the token id that divides input context and target context
    inputs, labels = [], []
    for seq in batch:
        idx=seq.index(sep_id)
        inputs.append(seq[:idx+1])
        labels.append(seq[idx+1:])
    return [inputs, labels]

def padSeqs(sequences, pad_id, maxlen=None):
    lengths = [len(x) for x in sequences]
    maxlen=max(lengths)
    maxlen=min(512, maxlen)
    
    pad_batch=np.ones((len(sequences), maxlen))*pad_id
    for idx, s in enumerate(sequences):
        trunc = s[-maxlen:]
        pad_batch[idx, :len(trunc)] = trunc
            
    return pad_batch

def integrate_result(inputs, gens, oracles):
    results=[]
    for context, gen, oracle in zip(inputs, gens, oracles):
        EN_list, left=context.split('[EOS_L]')
        EN_list=EN_list.strip('[CLS]')
        user, left = left.split('[EOS_U]')
        EN, left=left.split('[EOS_E]')
        user_intent, KB_result=left.split('[EOS_UI]')
        KB_result=KB_result.strip('[EOS_K]')
        service_intent, service=oracle.split('[EOS_SI]')
        service=service[:service.index('[EOS_S]')]
        if '[EOS_SI]' in gen:
            service_intent_gen, service_gen=gen.split('[EOS_SI]')
            service_gen=service_gen[:service_gen.index('[EOS_S]')]
        else:
            service_intent_gen=''
            service_gen=gen[:gen.index('[EOS_S]')]
        entry={
            '用户':user.replace(' ', ''),
            '用户意图':user_intent.replace(' ', ''),
            '实体列表':EN_list.replace(' ', ''),
            '实体':EN.replace(' ', ''),
            '数据库结果':KB_result.replace(' ', ''),
            '客服':service.replace(' ', ''),
            '客服意图':service_intent.replace(' ', ''),
            '客服-生成':service_gen.replace(' ', ''),
            '客服意图-生成':service_intent_gen.replace(' ', '')
        }

        results.append(entry)
    return results

def batch_align(contexts,left_len=0,return_attn=False): # left padding
    max_len=max([len(context) for context in contexts])
    max_len=min(512, max_len) # 1024-left_len
    new_contexts=[]
    attentions=[]
    for id, context in enumerate(contexts):
        if len(context)<max_len:
            new_context=(max_len-len(context))*[cfg.pad_id]+context
            attention=(max_len-len(context))*[0]+len(context)*[1]
        else:
            new_context=context[-max_len:]
            attention=len(new_context)*[1]
        new_contexts.append(new_context)
        attentions.append(attention)
    if return_attn:
        return new_contexts, attentions
    return new_contexts

def get_kb(retrieval_model):
    kb = {}
    texts = []
    answers = []
    KB = json.load(open('qa_cleaned_new.json','r'))
    for q,a in KB.items():
        texts.append(q)
        answers.append(a)
    embs = retrieval_model.encode(texts)
    return embs, answers

def get_api(retrieval_model, qa=True):
    texts = []
    answers = []
    API = all_api_intents
    if not qa:
        for q,a in API.items():
            texts.append(a) # texts.append(q)
            answers.append(q)
    else:
        for q,a in API.items():
            if q!="[QA]":
                texts.append(a) # texts.append(q)
                answers.append(q)
        KB = json.load(open('qa_cleaned_new.json','r'))
        for q,a in KB.items():
            texts.append(q)
            answers.append(a)
    embs = retrieval_model.encode(texts)
    return embs, answers

def get_spoken(spoken, new_input, role):
    result =[]
    hists = []
    role_shift = {'user':' 用户：', 'system':' 客服：'}
    for i in range(len(new_input)):
        s = ((spoken[i] if spoken!=[] else '') + role_shift[role] + new_input[i]).replace('[EOS_K]','').replace('[EOS_UI]','').replace('[EOS_SI]','').replace('[EOS_L]','').replace('[UNK]','')
        turns = s.split(' 用户：')
        if len(turns)> cfg.retrieve_hist + 1:
            hist = ' 用户：' + (' 用户：').join(turns[-cfg.retrieve_hist:])
        else:
            hist = s
        result.append(s)
        hists.append(hist)
    return result, hists

def get_new_qa(return_result=False):
    files= ['data/dev_final_processed.json', 'data/train_final_processed.json'] #, 'data/test_final.json']
    kb = {}
    kb_ret = {}
    ret_kb = json.load(open('data/global_kb.json','r')) 
    KB_org = json.load(open('qa.json','r'))
    for q,a in KB_org.items():
        kb[q] = a
        descript = f"问题：{q}, 答案：{a}"
        ret_kb.append(descript)
        #kb_ret[q] = descript
    for file in files:
        data = json.load(open(file, 'r'))
        for id, dial in data.items():
            hist = ['']
            #for turn in dial:
            for turn in dial['log']:
                _ , hist = get_spoken(hist, [clean_utt(turn['user'].lower())], role='user')
                if turn['api_query'] == '[QA]':
                    tmp_hist = copy.deepcopy(hist[0])
                    system_cleaned = clean_utt(turn['system'].lower())
                    kb[tmp_hist] = system_cleaned
                    descript = f"问题：{tmp_hist}, 答案：{system_cleaned}"
                    kb_ret[system_cleaned] = descript
                    #ret_kb.append(tmp_hist)
                    ret_kb.append(descript)
                _ , hist = get_spoken(hist, [clean_utt(turn['system'].lower())], role='system')
    if return_result:
        return kb, kb_ret, ret_kb
    else:
        json.dump(kb, open('qa_cleaned_new.json', 'w'), indent=2, ensure_ascii=False)
        json.dump(kb_ret, open('qa_cleaned_ret.json', 'w'), indent=2, ensure_ascii=False)
        json.dump(ret_kb, open('kb/ret_qa.json', 'w'), indent=2, ensure_ascii=False)
        return 

def analysis_data():
    all_apis = {}
    dialogs = {}
    files= ['data/dev.json', 'data/train.json', 'data/test.json']
    for file in files:
        data = json.load(open(file, 'r'))
        for id, dial in data.items():
            dialogs[id] = dial
    multi_query_turns = []
    avoid = ['取消', '办理', '验证']
    qa_num = 0
    api_num = 0
    special_api = []
    for id, dialog in dialogs.items():
        for turn in dialog:
            if turn['api_query'] =='[QA]':
                qa_num += 1
            elif turn['api_query'] !='':
                api_num += 1
                api = clean_intent(turn['api_query'])
                if api not in all_apis:
                    all_apis[api] = 0 
                all_apis[api] = all_apis[api] + 1
                if (turn['api_result']=='' and (turn['api_query'] not in avoid)) or len(turn['api_result'])>300:
                    special_api.append(turn)
    api = {i:all_apis[i] for i in sorted(all_apis, key=lambda x:all_apis[x], reverse=True)}
    print(f"qa_num:{qa_num}, api_num:{api_num}") 
    print(api)
    print(len(special_api))
    #json.dump(special_api, open("check_turn.json", 'w'), indent=2, ensure_ascii=False)
    return

if __name__=='__main__':
    #data = json.load(open(cfg.data_path,'r'))
    #get_retrieval_data(data)
    #clean_dataset_qa()
    get_new_qa()
    #analysis_data()

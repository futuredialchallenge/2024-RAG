import json
import logging
import os
import math
import numpy as np
import torch

from collections import OrderedDict
from tqdm import tqdm
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
#import ontology
import torch.nn as nn
from collections import OrderedDict
import random
import copy

#from process_annotated import dict_to_bspn

def get_sep_optimizers(cfg, num_turns, model, num_batches=None):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
    if not num_batches:
        num_training_steps = num_turns*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.batch_size)
    else:
        num_training_steps = num_batches*cfg.epoch_num
    #num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else
    num_warmup_steps = int(num_training_steps*cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,\
        num_training_steps=num_training_steps) #if cfg.use_scheduler else None
    logging.info('Training steps:{}, warmup steps:{}, steps per epoch:{}'.format(num_training_steps, 
        num_warmup_steps, num_batches))
    return optimizer, scheduler

def parse_arg_cfg(config,args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(config, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(config, k, v)
    return config

def padSeqs_gpt(sequences, pad_id, maxlen=None):
    lengths = []
    for x in sequences:
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_mexlen = np.max(lengths)

    # maxlen = 1024
    if seq_mexlen > 512: # gpt2.n_ctx
        # print('maxlen exceeds 1024')
        maxlen = 512
    else:
        maxlen = seq_mexlen

    # tokenizer.encode('<|endoftext|>') = ['50256']
    # All labels set to ``-100`` are ignored (masked), the loss is only
    # computed for labels in ``[0, ..., config.vocab_size]`` (from modeling_gpt2.GPT2LMHeadModel)
    
    x = (np.ones((num_samples, maxlen)) * pad_id)
    for idx, s in enumerate(sequences):
        if not len(s):
            print('empty list was found in padSeqs')
        # trunc method = 'pre'
        trunc = s[-maxlen:]
        trunc = np.asarray(trunc)

        # pad method = 'post'
        x[idx, :len(trunc)] = trunc
            
    return x, lengths
    
def generate_batch(self, encoded_data,model):
    with torch.no_grad():
        input=encoded_data

def rewrite():
    data=json.loads(open('data/result.json', 'r', encoding='utf-8').read())
    json.dump(data, open('data/results.json', 'w'), indent=2,ensure_ascii=False)
    return(0)

def aspn_to_response(domain,act,num=3):
    with open('ontology.json','r') as f:
         responses=json.loads(f.read())['act_response']
    response=[]
    if domain in responses:
        if act in responses[domain]:
            response=random.choices(responses[domain][act],k=num)
    else:
        if act=='reqmore':
            response=random.choices([responses['回访'][act][2],responses['回访'][act][4]],k=num)
        if act=='bye':
            response=random.choices(responses['回访'][act],k=num)
        if '询问' in act:
            proposal=act.replace('-','').replace('询问','请问一下')+'是什么呢？'
            response=random.choices(proposal,k=num)
    return response

def rule_based(first_turn,state,d_gen,bspn_gen,req_id=False):
    p_end_none=0.5
    p_end=0.5
    p_reqmore=0.3 
    end_domain = ['广告','产品']
    with open('ontology.json','r') as f:
        ontology=json.loads(f.read())
    schema = ontology['ontology']
    remain_slot=[]
    if d_gen in schema:
        require_slot = schema[d_gen]
         
        for slot in require_slot:
            if (slot in bspn_gen) or (slot in state['act']):
                if slot not in state['mentioned']:
                    state['mentioned'].append(slot)
        for slot in require_slot:
            if slot not in state['mentioned']:
                remain_slot.append(slot)
    else:
        d_gen = '未知'
    tmp1 = random.random()
    tmp2 = random.random()
    if d_gen=='未知':
         a_gen='reqmore' if state['act']=='' else 'bye'
    elif remain_slot==[]:
        if state['act']=='reqmore':
            a_gen='bye'
        elif tmp1<p_end_none:
            a_gen='bye'
        else:
            a_gen='reqmore'
    elif d_gen in end_domain:
        if state['act']=='reqmore':
            a_gen='bye'
        else:
            slot_asked=random.sample(remain_slot,1)[0]
            if tmp2<p_end :
                a_gen='bye'
            if tmp2<(p_end+p_reqmore) :
                a_gen='reqmore'
            else:
                a_gen='询问-'+slot_asked
    else:
        slot_asked=random.sample(remain_slot,1)[0]
        a_gen='询问-'+slot_asked
    state['remain'] = remain_slot
    state['act'] = a_gen
    if aspn_to_response(d_gen,a_gen,1)!=[]:
        r_gen=aspn_to_response(d_gen,a_gen,1)[0]
    else:
        a_gen = 'bye'
        if aspn_to_response(d_gen,a_gen,1)!=[]:
            r_gen=aspn_to_response(d_gen,a_gen,1)[0]
        else:
            r_gen='好嘞，我会替你给Ta转达的，那先这样，再见！'
    
    return state,a_gen,r_gen

def process_bspn_simple(bspn, domain , last_state=None):
    schema = {
        "娱乐":["地点", "时间"],
        "面试":["地点", "时间", "公司", "岗位"],
        "饭局":["地点", "时间"],
    }
    requested_slots = schema[domain] if domain in schema else ["地点", "时间", "公司"]
    bs = {}
    state = copy.deepcopy(last_state)
    for slot in requested_slots:
        if (slot in bspn) and (last_state[slot]==0 if (slot in last_state) else True):
            state[slot] = 1
            bs[slot] = 1
        else:
            bs[slot] = 0
    return bs, state

def process_bspn_single(bspn, domain):
    schema = {
        "娱乐":["地点", "时间"],
        "面试":["地点", "时间", "公司", "岗位"],
        "饭局":["地点", "时间"],
    }
    requested_slots = schema[domain] if domain in schema else ["地点", "时间", "公司"]
    bs = {}
    for slot in requested_slots:
        if slot in bspn:
            bs[slot] = 1
        else:
            bs[slot] = 0
    return bs

def process_bspn(bspn_gen,caller,bs_last=None):
    new_bspn = {}
    all_slots = []
    schema = json.loads(open('data/new_ontology.json', 'r', encoding='utf-8').read())#only domain and slots used, to be renewed
    for d,slots in schema['bspn'].items():
        for slot in slots:
            if slot not in all_slots:
                all_slots.append(slot)
    bs_tokens = bspn_gen.replace('[CLS]','').replace('<sos_b>','').replace('<eos_b>','').replace('[SEP]','').split(',')
    slot=''
    for token in bs_tokens:
        if token !='':
            if (token in all_slots) and (token not in new_bspn):
                slot = token
                new_bspn[slot] = ''
            elif slot!='':
                if bs_last:
                    if (token not in new_bspn[slot]) and ((token in caller) or (token in bs_last)):
                        new_bspn[slot] = token
                elif (token not in new_bspn[slot]) and (token in caller):
                    new_bspn[slot] = token

    return new_bspn

def process():
    files = ['data/train_new.json', 'data/eval_new.json', 'data/test_new.json']
    special = []
    for file in files:
        data = json.load(open(file,'r'))
        for dial in data:
            for turn in dial['log']:
                if 'domain' in turn:
                    if turn['domain'] == '0' or turn['domain'] == '':
                        dial['log'].remove(turn)
                        special.append(turn)
        json.dump(data, open(file, 'w'), indent=2, ensure_ascii=False)  

def dict_to_bspn(belief_state):
    banned_slot = ['事情', '产品', '特点', '方式', '金额', '价格']
    bspn=''
    for slot,value in belief_state.items():
        if value!='' and slot not in banned_slot: # can be changed into banned slots later
            if bspn=='': 
                bspn = slot + ' ' + (value if '/' not in value else value.split('/')[0])
            else: 
                bspn = bspn + ' ' + slot + ' ' + (value if '/' not in value else value.split('/')[0])
    return bspn 


def modify():
    files = ['data/train_new.json', 'data/eval_new.json', 'data/test_new.json']
    special = []
    blur_slot = ['公司', '平台', '机构']
    for file in files:
        data = json.load(open(file,'r'))
        for dial in data:
            if dial['annotation'] == 'v1':
                belief_state = {}
                for turn in dial['log']:
                    for k,v in turn['bspn'].items():
                        if v!='':
                            value = v.split('/')[0] if '/' in v else v
                            key = k if k not in blur_slot else '公司'
                            if key not in belief_state:
                                belief_state[key]=value.replace(',',' ')
                            else:
                                if value not in belief_state[key]:
                                    belief_state[key] = value.replace(',',' ')
                    turn['bstring']=dict_to_bspn(belief_state)
        json.dump(data, open(file, 'w'), indent=2, ensure_ascii=False)

def get_aug_ents():
    slots = {'公司':'data/other/company.txt', '岗位':'data/other/post.txt'}
    ents = {}
    ent_path = 'data/other/ents.json'
    for slot,file in slots.items():
        ents[slot] = []
        tmp = '1'
        with open(file, "r", encoding="utf-8") as fp:
            while tmp != "":
                tmp = fp.readline()
                if tmp != '':
                    tmp_list = tmp.split('\t')
                    ents[slot].append(tmp.replace('\n', ''))
        fp.close()
    json.dump(ents, open(ent_path, 'w'), ensure_ascii=False)
    return

def get_relevant_data():
    relevant_domain = ['面试', '饭局', '娱乐']
    relevant_slots = ['时间', '地点', '岗位'] # '公司', '平台', '机构'
    slot_switch = {'期限': '时间', '平台': '公司', '机构': '公司'}
    train_data = json.load(open('data/train_new.json', 'r'))
    new_dials = []
    for dial in train_data:
        flag = 0 
        turn = dial['log'][-1] if len(dial['log'])>0 else []
        if 'bstring' in turn:
            for s in relevant_slots:
                if s in turn['bstring']:
                    flag = 1
        if 'domain' in turn:
            for d in relevant_domain:
                if d in turn['domain']:
                    flag = 1
        if flag == 1:
            new_dials.append(dial)
    for dialog in new_dials:
        belief_state={}
        for turn in dialog['log']:
            if 'bspn' in turn:
                for k,value in turn['bspn'].items():
                    if value!='':
                        key = slot_switch[k] if k in slot_switch else k
                        if key not in belief_state:
                            belief_state[key] = value.replace(',',' ')
                        else:
                            belief_state[key] = value.replace(',',' ')
                turn['bstring']=dict_to_bspn(belief_state)
    json.dump(new_dials, open('data/train_relevant.json', 'w'), indent=2, ensure_ascii=False)
    return

def get_relevant_data1():
    data = json.load(open('ex/data_processed.json', 'r', encoding='utf-8'))
    for dialog in data:
        belief_state={}
        for turn in dialog['log']:
            if 'bspn' in turn:
                for key,value in turn['bspn'].items():
                    if value!='':
                        if key not in belief_state:
                            belief_state[key] = value.replace(',',' ')
                        else:
                            belief_state[key] = value.replace(',',' ')
                turn['bstring']=dict_to_bspn(belief_state)
    json.dump(data, open('ex/data_processed.json', 'w'), indent=2, ensure_ascii=False)

def analysis_relevant_data():
    data = json.load(open('data/train_relevant.json', 'r'))
    domains = {}
    turns = []
    for dialog in data:
        belief_state={}
        for turn in dialog['log']:
            if 'bspn' in turn:
                for key,value in turn['bspn'].items():
                    if value!='':
                        if key not in belief_state:
                            belief_state[key] = value.replace(',',' ')
                        else:
                            belief_state[key] = value.replace(',',' ')
                turn['bstring']=dict_to_bspn(belief_state)
    json.dump(data, open('data/train_relevant.json', 'w'), indent=2, ensure_ascii=False)
    return

def dict_to_turn_bspn(belief_state):
    banned_slot = ['事情', '产品', '特点', '方式', '金额', '价格']
    slot_switch = {'期限': '时间', '平台': '公司', '机构': '公司'}
    bspn=''
    for slot,value in belief_state.items():
        if value!='' and slot not in banned_slot: # can be changed into banned slots later
            if '/' in value:
                v = value.split('/')[0] 
            elif ',' in value:
                v = value.split(',')[0]
            else:
                v = value
            if bspn=='': 
                bspn = (slot if slot not in slot_switch else slot_switch[slot]) + ' ' + v   
            else: 
                bspn = bspn + ' '+ (slot if slot not in slot_switch else slot_switch[slot]) + ' ' + v
    return bspn

def get_turn_data1():
    #files = ['ex/data_processed.json', 'data/train_augmented.json', 'ex/test_processed_new.json'] # 'data/files= train_relevant.json'
    files = ['data/train_turn_old.json']
    aug_ents = json.load(open('data/other/ents.json', 'r'))
    for file in files:
        data = json.load(open(file, 'r'))
        domains = {}
        turns = []
        for dial in data:
            for num in range(len(dial['log'])):
                turn = dial['log'][num]
                if 'bspn' in turn:
                    turn['bstring'] = dict_to_turn_bspn(turn['bspn'])
                else:
                    turn['bstring'] = ''
        json.dump(data, open('data/train_turn.json', 'w'), indent=2, ensure_ascii=False) # 'data/train_turn.json'
        # file
    return

def get_turn_data():
    #files = ['ex/data_processed.json', 'data/train_augmented.json', 'ex/test_processed_new.json'] # 'data/files= train_relevant.json'
    files = ['ex/test_20221228_new.json']
    aug_ents = json.load(open('data/other/ents.json', 'r'))
    for file in files:
        data = json.load(open(file, 'r'))
        domains = {}
        turns = []
        new_data = []
        for id, dial in data.items():
            for num in range(len(dial['log'])):
                turn = dial['log'][num]
                turn['domain'] = turn['domain_new']
                if turn['domain'] == '兜底场景':
                    turn['domain'] = '未知'
                if turn['domain'] == '其他场景':
                    turn['domain'] = '其他'
                if 'bspn' in turn:
                    tmp_bspn = turn['bspn'][turn['domain_new']] if turn['domain_new'] in turn['bspn'] else {}
                    bspn = {}
                    for k,v in tmp_bspn.items():
                        if v!='':
                            value = v.split('/')[0] if '/' in v else v
                            key = k.split('/')[0] if '/' in k else k
                            bspn[key] = value
                    turn['bstring'] = dict_to_turn_bspn(bspn)
                else:
                    turn['bstring'] = ''
                    turn['bspn'] = {}
                turn['bspn'] = bspn
            new_data.append(dial)
        json.dump(new_data, open('ex/test_processed_new1.json', 'w'), indent=2, ensure_ascii=False) # 'data/train_turn.json'
        # file
    return

def get_aug_data():
    #files = ['ex/data_processed.json', 'data/train_augmented.json', 'ex/test_processed_new.json'] # 'data/files= train_relevant.json'
    #files = ['data/train_turn_old.json']
    files = ['data/bad_case.json']
    aug_ents = json.load(open('data/other/ents.json', 'r'))
    #AUG_NUM = 2
    #the method of augmenting needs to be refined
    for file in files:
        data = json.load(open(file, 'r'))
        domains = {}
        turns = []
        aug = []
        for dial in data:
            new_dial = copy.deepcopy(dial)
            new_dial1 = copy.deepcopy(dial)
            outer_flag = 0
            for num in range(len(dial['log'])):
                turn = dial['log'][num]
                if 'bspn' in turn:
                    flag = 0
                    turn['bstring'] = dict_to_turn_bspn(turn['bspn'])
                    if '公司' in turn['bspn']:
                        tmp = turn['bspn']['公司']
                        if tmp in turn['caller'] and tmp!='':
                            n = random.sample(aug_ents['公司'], 1)[0]
                            new_dial['log'][num]['caller'] = new_dial['log'][num]['caller'].replace(tmp, n)
                            new_dial['log'][num]['bspn']['公司'] = n

                            n = random.sample(aug_ents['公司'], 1)[0]
                            new_dial1['log'][num]['caller'] = new_dial1['log'][num]['caller'].replace(tmp, n)
                            new_dial1['log'][num]['bspn']['公司'] = n

                            flag = 1
                    if '岗位' in turn['bspn']:
                        tmp = turn['bspn']['岗位']
                        if tmp in turn['caller']:
                            n = random.sample(aug_ents['岗位'], 1)[0]
                            new_dial['log'][num]['caller'] = new_dial['log'][num]['caller'].replace(tmp, n)
                            new_dial['log'][num]['bspn']['岗位'] = n

                            n = random.sample(aug_ents['岗位'], 1)[0]
                            new_dial1['log'][num]['caller'] = new_dial1['log'][num]['caller'].replace(tmp, n)
                            new_dial1['log'][num]['bspn']['岗位'] = n

                            flag = 1
                    if flag==1:
                        new_dial['log'][num]['bstring'] = dict_to_turn_bspn(new_dial['log'][num]['bspn'])
                        new_dial1['log'][num]['bstring'] = dict_to_turn_bspn(new_dial1['log'][num]['bspn'])
                        outer_flag = 1
                else:
                    turn['bstring'] = ''
            if outer_flag==1:
                aug.append(new_dial)
                aug.append(new_dial1)
        print(len(aug))
        data.extend(aug)
        #json.dump(data, open('data/train_turn.json', 'w'), indent=2, ensure_ascii=False) # 'data/train_turn.json'
        json.dump(data, open('data/bad_case_augmented.json', 'w'), indent=2, ensure_ascii=False)
        # file
    return

def add_data():
    origin_data = json.load(open('data/train_turn.json', 'r'))
    origin_data.extend(json.load(open('data/bad_case_augmented.json', 'r')))
    json.dump(origin_data, open('data/train_turn_v1.json', 'w'), indent=2, ensure_ascii=False)

def check():
    #file = 'data/train_turn.json'
    file = 'data/train_new.json'
    data = json.load(open(file, 'r'))
    special = []
    for dial in data:
        for num in range(len(dial['log'])):
            turn = dial['log'][num]
            if 'bspn' in turn:
                if '公司' in turn['bspn']:
                    if turn['bspn']['公司'] not in turn['caller']:
                        special.append(turn)
                if '岗位' in turn['bspn']:
                    if turn['bspn']['岗位'] not in turn['caller']:
                        special.append(turn)
    json.dump(data, open('data/train_turn_v1.json', 'w'), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # rule_based('未知','1')
    # process()
    # modify()
    # get_aug_ents()
    # get_relevant_data()
    # get_relevant_data1()
    # analysis_relevant_data()
    #get_turn_data()
    #get_aug_data()
    #add_data()
    check()



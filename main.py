"""
Copyright 2024 Tsinghua University
Author: Yucheng Cai (cyc22@mails.tsinghua.edu.cn)
"""

import os, shutil
import random
import argparse
import time
import logging
import json
import copy, re

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel, BertModel, BertForNextSentencePrediction, MT5ForConditionalGeneration
from transformers import BertTokenizer, T5Tokenizer
from sentence_transformers import models, SentenceTransformer, losses, datasets, util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from reader import *
from metrics import *
from config import global_config as cfg

from retrieve_kb import Api_model, collate_fn, Retrieval_model, build_global_index, build_local_index

class Model(object):
    def __init__(self, device=[0],posterior = False):
        self.posterior = posterior
        if isinstance(device,list):
            self.device = device[0]
        else:
            self.device = device
        if posterior:
            self.model=GPT2LMHeadModel.from_pretrained(cfg.posterior_path) if cfg.gpt else MT5ForConditionalGeneration.from_pretrained(cfg.t5_posterior_path)
            self.tokenizer = BertTokenizer.from_pretrained(cfg.posterior_path) if cfg.gpt else BertTokenizer.from_pretrained(cfg.t5_posterior_path)
        else:
            self.model=GPT2LMHeadModel.from_pretrained(cfg.gpt_path) if cfg.gpt else MT5ForConditionalGeneration.from_pretrained(cfg.t5_path)
            self.tokenizer = BertTokenizer.from_pretrained(cfg.gpt_path) if cfg.gpt else BertTokenizer.from_pretrained(cfg.t5_path)
        self.model.to(self.device)

        if 'train' in cfg.mode:
            json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
        
            # Add special tokens
            init_vocab_size=len(self.tokenizer)
            special_tokens_dict = {'additional_special_tokens': special_tokens}
            logging.info('Added special tokens:{}'.format(special_tokens))
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
            logging.info('Special token added, vocab size:{}-->{}'.format(init_vocab_size, len(self.tokenizer)))

        # log
        log_path='./log/log_{}'.format(cfg.exp_name)
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
            os.mkdir(log_path)
        else:
            os.mkdir(log_path)
        self.tb_writer = SummaryWriter(log_dir=log_path)
    
    def train(self):
        if not cfg.only_target_loss:
            if (cfg.mode == 'train' or cfg.mode == 'train_post'):
                encoded_data=read_data(self.tokenizer,self.posterior)
            if cfg.mode == 'pretrain':
                encoded_data=get_unsup(self.tokenizer,pretrain=True)
            train_dataloader=DataLoader(encoded_data['train'], batch_size=cfg.batch_size, shuffle=True, collate_fn=train_collate_fn) 
            dev_dataloader=DataLoader(encoded_data['dev'], batch_size=cfg.eval_batch_size, collate_fn=train_collate_fn)
            optimizer, scheduler = self.get_optimizers(len(encoded_data['train']), self.model)
        else:
            encoded_data=read_data(self.tokenizer,return_dict=True)
            train_data=encoded_data['train'] # [:400]
            random.shuffle(train_data)
            train_dataloader = get_batches(train_data,cfg.batch_size)
            #num_dials=len(train_data)
            num_turns = sum(len(dial) for dial in train_data)
            optimizer, scheduler = self.get_optimizers(num_turns, self.model)
            cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
            sep = self.tokenizer.convert_tokens_to_ids('[SEP]')
            qa = self.tokenizer.convert_tokens_to_ids('[QA]')

        if cfg.debugging:
            train_dataloader = train_dataloader[:len(train_dataloader)//40]
        log_inputs = 2
        global_step = 0
        min_loss=10000
        max_score=0
        for epoch in range(cfg.epoch_num):
            tr_loss = 0.0
            step_loss = 0
            epoch_step = 0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()
            for batch_idx, batch in enumerate(train_dataloader):
                self.model.train()
                try:  # avoid OOM
                    if cfg.only_target_loss: # default setting
                        dial_batch=transpose_batch(batch)
                        pv_batch = None
                        spoken = []
                        for turn_num, turn_batch in enumerate(dial_batch):
                            first_turn = (turn_num == 0)
                            if cfg.gpt:
                                inputs, labels = convert_batch_turn(config=cfg, cls=cls, sep=sep, 
                                turn_batch=turn_batch, pv_batch=pv_batch, 
                                first_turn=first_turn, posterior=self.posterior, qa=qa) 
                                # previously, pv_batch is not used in training, using ground truth
                                # now, pv_batch means last_turn
                                pv_batch = self.get_pv_batch(pv_batch, turn_batch['user'], turn_batch['resp'])
                                if log_inputs > 0:  # log inputs for the very first two turns
                                    logging.info('Input examples:\n{}'.format(self.tokenizer.decode(inputs['contexts'][0])))
                                    log_inputs-=1
                                inputs = self.add_torch_input(inputs)
                                labels = self.add_torch_input(labels) 
                                outputs = self.model(inputs['contexts_tensor'])
                                #if cfg.mix_retrieval_training:  
                                #    loss = loss + self.calculate_loss_and_accuracy(outputs, labels=labels['contexts_tensor']) 
                                #else:
                                loss = self.calculate_loss_and_accuracy(outputs, labels=labels['contexts_tensor'])
                            else:
                                inputs, labels = convert_batch_t5(cls,sep,turn_batch, turn_batch['ent_list'], first_turn, posterior=self.posterior)
                                pv_batch = self.get_pv_batch(pv_batch, turn_batch['user'], turn_batch['resp'])
                                if log_inputs > 0:  # log inputs for the very first two turns
                                    logging.info('Input examples:\n{}'.format(self.tokenizer.decode(inputs['contexts'][0])))
                                    logging.info('Output examples:\n{}'.format(self.tokenizer.decode(labels['contexts'][0])))
                                    log_inputs-=1
                                inputs = self.add_torch_input(inputs)
                                labels = self.add_torch_input(labels)
                                labels['contexts_tensor'][labels['contexts_tensor'] == self.tokenizer.pad_token_id] = -100
                                loss = self.model(input_ids=inputs['contexts_tensor'], attention_mask=inputs['attention_tensor'], labels=labels['contexts_tensor'], return_dict=False)[0]
                            loss.backward()
                            tr_loss += loss.item()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                            epoch_step += 1

                            # step, wrt gradient_accumulation_steps, clip grad norm
                            if epoch_step % cfg.gradient_accumulation_steps == 0 or(
                                batch_idx==len(train_dataloader)-1 and turn_num==len(dial_batch)-1):
                                optimizer.step()
                                scheduler.step()
                                optimizer.zero_grad()
                                # global_step: actual step the optimizer took
                                global_step += 1
                    else:
                        if log_inputs > 0:  # log inputs for the very first two turns
                            logging.info('Training Sequences:')
                            if cfg.gpt:
                                logging.info(self.tokenizer.decode(batch[0,:]))
                            else:
                                logging.info(self.tokenizer.decode(batch['input'][0,:]))
                            log_inputs-=1
                        if cfg.gpt:
                            inputs=batch.to(self.device) #B, T
                            labels=inputs
                            outputs = self.model(inputs)
                            loss = self.calculate_loss_and_accuracy(outputs, labels=labels)
                        else:
                            for key, val in batch.items():
                                if type(batch[key]) is list:
                                    continue
                                batch[key] = batch[key].to(self.device)
                            batch['output'][batch['output'] == self.tokenizer.pad_token_id] = -100
                            loss = self.model(input_ids=batch['input'], labels=batch['output'], return_dict=False)[0]
                        loss=loss/cfg.gradient_accumulation_steps
                        loss.backward()
                        tr_loss += loss.item()
                        step_loss+=loss.item()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                        if (batch_idx+1) % cfg.gradient_accumulation_steps == 0 or batch_idx+1==len(train_dataloader):
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            global_step += 1
                            self.tb_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], global_step)
                            self.tb_writer.add_scalar('loss', step_loss, global_step)
                            step_loss=0

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        oom_time += 1
                        logging.info("WARNING: ran out of memory,times: {}, batch size: {}".format(oom_time, cfg.batch_size))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        logging.info(str(exception))
                        raise exception
            if not cfg.only_target_loss:
                eval_loss=self.eval(dev_dataloader)
            if cfg.save_type =='max_score':
                if self.posterior:
                    eval_result=self.validate_post()
                    ui = eval_result['P/R/F1 for user intent'][2]
                    db = eval_result['P/R/F1 for db prediction'][2] 
                    eval_loss = ui + si + db
                    logging.info('user:{:.3f}, system:{:.3f} , db:{:.3f}, score:{:.3f}'.format(ui, si, db, eval_loss))
                else:
                    eval_result=self.validate_fast('test') # 'test'
                    bleu = eval_result['BLEU']
                    #f1_query = eval_result['query_f1'][2]
                    #qa = eval_result['qa'][2]
                    inform = eval_result['inform']
                    bert_score = eval_result['bert_score'].item()
                    eval_loss = bleu/50 + inform + bert_score
            logging.info('Epoch:{}, Train epoch time:{:.2f} min, epoch loss:{:.3f}, eval loss:{:.3f}'.format(epoch, (time.time()-btm)/60, tr_loss, eval_loss))
            self.tb_writer.add_scalar('eval_loss', eval_loss, epoch)
            if cfg.save_type =='max_score':
                if max_score < eval_loss:
                    max_score=eval_loss
                    self.save_model()
            else:
                if eval_loss<min_loss:
                    min_loss=eval_loss
                    self.save_model()
        #self.save_model(path='last_epoch_model')

    def train_one_step(self,batches,optimizer,scheduler):
        tr_loss = 0.0
        step_loss=0
        oom_time = 0
        for batch_idx, batch in enumerate(batches):
            try:  # avoid OOM
                self.model.train()
                #if log_inputs > 0:  # log inputs for the very first two turns
                #    logging.info('Training Sequences:')
                #    logging.info(self.tokenizer.decode(batch[0,:]))
                #    log_inputs-=1
                inputs=batch.to(self.device) #B, T
                labels=inputs
                outputs = self.model(inputs)
                loss = self.calculate_loss_and_accuracy(outputs, labels=labels)
                loss=loss/cfg.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                step_loss+=loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                if (batch_idx+1) % cfg.gradient_accumulation_steps == 0 or batch_idx+1==len(batches):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    #self.tb_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], global_step)
                    #self.tb_writer.add_scalar('loss', step_loss, global_step)
                    step_loss=0
                    #torch.cuda.empty_cache()
                    #schedule_count = 0
                #else:
                    #schedule_count = schedule_count + 1

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    logging.info("WARNING: ran out of memory,times: {}, batch size: {}".format(oom_time, cfg.batch_size))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logging.info(str(exception))
                    raise exception
        return tr_loss

    def get_optimizers(self, num_samples, model):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        #print(num_samples, cfg.epoch_num, cfg.gradient_accumulation_steps, cfg.batch_size)
        num_training_steps = num_samples*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.origin_batch_size) # origin_batch_size_here
        num_warmup_steps = int(num_training_steps*cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,\
            num_training_steps=num_training_steps)
        return optimizer, scheduler
    
    def calculate_loss_and_accuracy(self, outputs, labels):
        # GPT2-chicahat/train.py
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=cfg.pad_id, reduction='sum')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(cfg.pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss
    
    def eval(self, data):
        self.model.eval()
        total_loss=0
        with torch.no_grad():
            for batch in data:
                if cfg.gpt:
                    inputs=batch.to(self.device) #B, T
                    labels=inputs
                    outputs = self.model(inputs)
                    loss = self.calculate_loss_and_accuracy(outputs, labels=labels)
                else:
                    for key, val in batch.items():
                        if type(batch[key]) is list:
                            continue
                        batch[key] = batch[key].to(self.device)
                    batch['output'][batch['output'] == self.tokenizer.pad_token_id] = -100
                    loss = self.model(input_ids=batch['input'], labels=batch['output'], return_dict=False)[0]
                total_loss+=loss.item()
        return total_loss/len(data)

    def save_model(self, path=None, model=None):
        if self.posterior:
            save_path = os.path.join(cfg.exp_path, path) if path else os.path.join(cfg.exp_path, 'best_post_model')
        else:
            save_path = os.path.join(cfg.exp_path, path) if path else os.path.join(cfg.exp_path, 'best_model')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        if not model:
            self.model.save_pretrained(save_path)
        else:
            model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def add_torch_input(self, inputs):
        # to tensor and to device
        if 'contexts_np' not in inputs:
            inputs['contexts_np'],_=padSeqs_gpt(inputs['contexts'], cfg.pad_id)
        # add attention if attention in inputs
        if 'attention' in inputs:
            attentions_tensor = torch.from_numpy(inputs['attention']).long()
            attentions_tensor = attentions_tensor.to(self.device)
            inputs['attention_tensor'] = attentions_tensor

        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        contexts_tensor = contexts_tensor.to(self.device)
        inputs['contexts_tensor'] = contexts_tensor
        return inputs

    def gen_hidden_state(self, turn_batch, pv_batch, posterior=True, validate=False):
        cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
        eos_a_id=self.tokenizer.convert_tokens_to_ids('[EOS_A]')
        eos_q_id=self.tokenizer.convert_tokens_to_ids('[EOS_Q]')
        sep_id=self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.model.eval()
        max_len_q=10
        max_len_a=50
        with torch.no_grad():
            # generate query
            contexts = convert_eval_batch_turn(config=cfg, cls=cls,  turn_batch=turn_batch, pv_batch=pv_batch,
             mode='gen_query', posterior=posterior)
            if cfg.gpt:
                query_batch=self.generate_batch(self.model, contexts, max_len_q, eos_q_id)
                query_gen = self.get_xspn(query_batch, eos_a_id,sep_id) 
            # generate answer
            contexts = convert_eval_batch_turn(config=cfg, cls=cls,  turn_batch=turn_batch, pv_batch=pv_batch,
             mode='gen_answer', posterior=posterior)
            if cfg.gpt:
                answer_batch=self.generate_batch(self.model, contexts, max_len_a, eos_a_id)
                answer_gen = self.get_xspn(answer_batch, eos_a_id,sep_id) 

            if validate:# generate hidden states for validation
                turn_batch['查询结果'] = self.decode_batch(turn_batch['api_result'])
                turn_batch['查询'] = self.decode_batch(turn_batch['api_query'])
                turn_batch['查询-生成'] = self.decode_batch(query_gen)
                turn_batch['查询结果-生成'] = self.decode_batch(answer_gen)
                turn_batch['客服'] = self.decode_batch(turn_batch['resp'])
                turn_batch['用户'] = self.decode_batch(turn_batch['user'])

            else:# generate hidden states for training
                turn_batch['api_query'] = query_gen
                turn_batch['api_result'] = answer_gen
                if 'user_decode' not in turn_batch:
                    turn_batch['user_decode'] = self.decode_batch(turn_batch['user'])
                    turn_batch['resp_decode'] = self.decode_batch(turn_batch['resp'])
            pv_batch = self.get_pv_batch(pv_batch, user = turn_batch['user'], resp = turn_batch['resp'])
        return turn_batch, pv_batch 

    def get_pv_batch(self, pv_batch, user = None, resp = None):#user = None
        # only reserves the last turn
        new_pv_batch=[] # pv_batch for next turn, return ent_list without any special tokens
        if pv_batch is None:# first turn
            for  u, r in zip( user, resp): 
                new_pv_batch.append(u + r) # remove eos_e
        else: 
            for hist, u, r in zip(pv_batch, user, resp):
                new_pv_batch.append(u + r)
        return new_pv_batch

    def get_xspn(self, input_tensor, eos_id, sep_id, sos_id = None):
        if not isinstance(input_tensor, list):
            input_batch=input_tensor.cpu().tolist()
        else:
            input_batch=input_tensor
        xspn_gen=[]
        for i ,xspn in enumerate(input_batch):
            if eos_id in xspn:
                xspn = xspn[:xspn.index(eos_id)+1]
            else:
                xspn[-1]=eos_id
            if sos_id:
                if sos_id in xspn:
                    xspn = xspn[xspn.index(sos_id)+1:] # multi sos_id not dealt with
                    if sos_id in xspn:
                        xspn = xspn[xspn.index(sos_id)+1:]
            xspn_new=[]
            for token in xspn:
                if token!=sep_id:
                    xspn_new.append(token)
            xspn_gen.append(xspn_new)
        return xspn_gen

    def query_kb_retrieval(self, passage_model, global_kb, KB_embed, api_model, local_kb, hist): 
        local_index = build_local_index(passage_model, local_kb, KB_embed)

        input = []
        attention_mask = []
        for h in hist:
            tokenized = self.tokenizer(h, max_length = 512, truncation=True)
            input.append(tokenized['input_ids'])
            attention_mask.append(tokenized['attention_mask'])

        max_len = max(len(i) for i in input)
        pad_input=np.ones((len(hist), max_len))*cfg.pad_id  #-100
        for idx, s in enumerate(input):
            pad_input[idx, :len(s)] = s
        pad_input = torch.from_numpy(pad_input).long()

        max_len = max(len(i) for i in attention_mask)
        pad_attention_mask=np.ones((len(hist), max_len))*cfg.pad_id  #-100
        for idx, s in enumerate(attention_mask):
            pad_attention_mask[idx, :len(s)] = s
        pad_attention_mask = torch.from_numpy(pad_attention_mask).long()

        pad_input = pad_input.to(cfg.device[0])
        pad_attention_mask = pad_attention_mask.to(cfg.device[0])

        logits = api_model(input_ids=pad_input, attention_mask=pad_attention_mask, indices=local_index,topk=1)

        KB_results=[]
        KB_seqs =[]
        for dial_num in range(len(hist)):
            kb = copy.deepcopy(global_kb)
            kb.extend(local_kb[dial_num])
            pred = logits[dial_num]
            predict_index = pred[1].cpu().tolist()
            KB_seq = kb[predict_index[0]]
            KB_results.append(self.tokenizer.encode(KB_seq+'[EOS_A]')[1:-1])
            KB_seqs.append(KB_seq)
        return KB_results, KB_seqs

    def query_api(self, q_gen, q_gt, api_result, context): #, retrieval_model
        KB_results=[]
        KB_seqs =[]
        for dial_num in range(len(q_gen)):
            # retrieval-based dialog, need specification
            #if q_gen[dial_num] == '[QA]':
            #    embed = retrieval_model(context[dial_num])
            #    KB_seq = embed
            if q_gen[dial_num] == q_gt[dial_num]:
                KB_seq = api_result[dial_num]
            else:
                KB_seq = ''
            KB_results.append(self.tokenizer.encode(KB_seq+'[EOS_A]')[1:-1])
            KB_seqs.append(KB_seq)
        return KB_results, KB_seqs

    def find_api(self, context, api_embed, api_answers, retrieval_model): 
        api_results=[]
        api_seqs =[]
        thresholds = [0.55, 0.55, 0.6, 0.55, 0.55]
        thresholds_qa = 0.9
        # 0.95 #  thresholds_qa with test in qa

        if cfg.dual_encoder:
            for dial_num in range(len(context)):
                embed = retrieval_model.encode(context[dial_num][-200:], show_progress_bar=False)
                scores = util.cos_sim(embed, api_embed)
                tmp_score = scores.cpu().tolist()
                if not cfg.retrieve_with_qa:
                    calibrated_score = [tmp_score[0][i]-thresholds[i] for i in range(len(thresholds))]
                    top = max(calibrated_score)
                    if top > 0.0:
                        api_seq = api_answers[calibrated_score.index(top)]
                    else:
                        api_seq = ''
                else:
                    tmp_score_qa = tmp_score[0][4:]
                    tmp_score_api = tmp_score[0][:4]
                    calibrated_score = [tmp_score_api[i]-thresholds[i] for i in range(len(thresholds)-1)]
                    top = max(calibrated_score)
                    top_qa = max(tmp_score_qa) - thresholds_qa
                    if top_qa>0.0:
                        api_seq='[QA]'
                    elif top > 0.0:
                        api_seq = api_answers[calibrated_score.index(top)]
                    else:
                        api_seq = ''
                api_results.append(self.tokenizer.encode(api_seq + '[EOS_Q]')[1:-1])
                api_seqs.append(api_seq)
                """
                flag = 0
                for i in range(len(thresholds)):
                    if scores[0, i].item()>thresholds[i]:
                        api_seq = api_answers[i]
                        flag = 1
                if flag == 0:
                    api_seq = ''
                """
        else:
            batch = []
            for i in range(len(context)):
                case = {}
                tokenized = self.tokenizer(context[i], max_length = 512, truncation=True)#, add_special_tokens=False)
                case['input'] = tokenized['input_ids']
                case['attention_mask'] = tokenized['attention_mask']
                batch.append(case)
            test_batch = collate_fn(batch)
            for key, val in test_batch.items():
                if type(test_batch[key]) is list:
                    continue
                test_batch[key] = test_batch[key].to(cfg.device[0])    
            logits = retrieval_model(input_ids=test_batch["input"],attention_mask=test_batch["attention_mask"])
            intent_preds = torch.argmax(logits, dim=1).cpu().tolist()
            for i in range(len(intent_preds)):
                api_seq = api_answers[intent_preds[i]]
                api_results.append(self.tokenizer.encode(api_seq + '[EOS_Q]')[1:-1])
                api_seqs.append(api_seq)
        return api_results, api_seqs

    def query_KB(self, q_gen, context, retrieval_model, KB_embed, KB_answer, resp_gen): 
        #, retrieval_model
        KB_results=[]
        KB_seqs =[]
        for dial_num in range(len(q_gen)):
            if q_gen[dial_num] == '[QA]':
                embed = retrieval_model.encode(context[dial_num], show_progress_bar=False)
                scores = util.cos_sim(embed, KB_embed)
                KB_seq = KB_answer[scores.argmax()]#index(max( scores )
                KB_results.append(self.tokenizer.encode(KB_seq + '[EOS_S]')[1:-1])
            else:
                KB_seq = ''
                KB_results.append(resp_gen[dial_num])
            KB_seqs.append(KB_seq)
        return KB_results, KB_seqs

    def decode_batch(self,result_encode,without_eos = True):
        result_decode =[]
        for encoded in result_encode:
            if without_eos:
                result_decode.append(self.tokenizer.decode(encoded[:-1]).replace(' ', '').replace('[CLS]', ''))
            else:
                result_decode.append(self.tokenizer.decode(encoded).replace(' ', '').replace('[CLS]', ''))
        return result_decode
    
    def get_spoken(self, spoken, new_input, role):
        result =[]
        hists = []
        role_shift = {'user':' 用户：', 'system':' 客服：'}
        for i in range(len(new_input)):
            s = ((spoken[i] if spoken!=[] else '') + role_shift[role] + new_input[i]).replace('[EOS_K]','').replace('[EOS_UI]','').replace('[EOS_SI]','').replace('[EOS_L]','').replace('[UNK]','')
            turns = s.split(' 用户：')
            if len(turns)> cfg.retrieve_hist:
                hist = ' 用户：' + (' 用户：').join(turns[-cfg.retrieve_hist:])
            else:
                hist = s
            result.append(s)
            hists.append(hist)
        return result, hists

    def encode_batch(self,batch,eos_id = None):
        result_encode =[]
        for sent in batch:
            if eos_id:
                result_encode.append(self.tokenizer.encode(sent)[1:-1] + [eos_id])
            else:
                result_encode.append(self.tokenizer.encode(sent)[1:-1])
        return result_encode

    def generate_batch_turn_level(self, batch, ground_truth_db=False, posterior=False,
    passage_model=None, KB_embed=None, KB_answers=None, api_model=None, api_embed=None,
    api_answers=None):
        cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
        eos_a_id=self.tokenizer.convert_tokens_to_ids('[EOS_A]')
        eos_q_id=self.tokenizer.convert_tokens_to_ids('[EOS_Q]')
        eos_r_id=self.tokenizer.convert_tokens_to_ids('[EOS_S]')
        sep_id=self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.model.eval()
        batch=transpose_batch(batch)
        max_len_q=20
        max_len_a=50
        max_len_resp=200
        batch_size=len(batch[0]['user'])
        contexts=[[] for i in range(batch_size)]
        resp_gen=[]
        pv_batch=None
        spoken = []

        if cfg.rag_training and cfg.rag_testing:
            global_kb = json.loads(open('kb/ret_qa.json', 'r').read())

        with torch.no_grad():
            # generate bspn
            for turn_num, turn_batch in enumerate(batch):
                user = self.decode_batch(turn_batch['user'])
                #print(len(spoken), len(user))
                spoken, hist = self.get_spoken(spoken, user, role='user')

                # generate q
                if cfg.rag_training or cfg.no_retrieval:
                    q_gen = turn_batch['api_query']
                    q_gt = self.decode_batch(turn_batch['api_query'])
                    q_decode = self.decode_batch(q_gen)
                elif cfg.only_response:
                    contexts = convert_eval_batch_turn(config=cfg, cls=cls,  turn_batch=turn_batch, pv_batch=pv_batch, 
                    mode='gen_query', posterior=posterior)
                    q_gen, q_decode = self.find_api(hist, api_embed, api_answers, api_model)
                    q_gt = self.decode_batch(turn_batch['api_query'])
                else:
                    contexts = convert_eval_batch_turn(config=cfg, cls=cls,  turn_batch=turn_batch, pv_batch=pv_batch, 
                    mode='gen_query', posterior=posterior)
                    q_batch=self.generate_batch(self.model, contexts, max_len_q, eos_q_id)
                    q_gen = self.get_xspn(q_batch, eos_q_id, sep_id)
                    q_decode = self.decode_batch(q_gen)
                    q_gt = self.decode_batch(turn_batch['api_query'])

                if cfg.rag_training and cfg.rag_testing:
                    a_gen, a_decode = self.query_kb_retrieval(passage_model, global_kb, KB_embed, api_model, turn_batch['local_kb'], hist)
                elif not cfg.gt_api:
                    a_gen, a_decode = self.query_api(q_decode, q_gt, a_decode, hist) # , best_model
                else:
                    a_gen = turn_batch['api_result']
                    a_decode = self.decode_batch(a_gen)

                contexts = convert_eval_batch_turn(config=cfg, cls=cls,  turn_batch=turn_batch, pv_batch=pv_batch
                , mode='gen_resp', q_gen=q_gen, a_gen=a_gen, posterior=posterior)    
                resp_batch=self.generate_batch(self.model, contexts, max_len_resp, eos_r_id)
                resp_gen = self.get_xspn(resp_batch, eos_r_id, sep_id)

                turn_batch['query_gen']=q_gen
                turn_batch['result_gen']=a_gen

                if not cfg.rag_training:
                    if (cfg.retrieve_qa and (not cfg.no_retrieval)):
                        resp_gen, resp_retrieve = self.query_KB(q_decode, hist, passage_model, KB_embed, KB_answers, resp_gen)
                turn_batch['resp_gen'] = resp_gen

                turn_batch['用户'] = user
                turn_batch['查询结果'] = self.decode_batch(turn_batch['api_result'])
                turn_batch['查询'] = self.decode_batch(turn_batch['api_query'])
                turn_batch['查询-生成'] = self.decode_batch(q_gen)
                turn_batch['查询结果-生成'] = a_decode# self.decode_batch(a_gen)
                turn_batch['客服'] = self.decode_batch(turn_batch['resp'])

                pv_batch = self.get_pv_batch(pv_batch, user = turn_batch['user'], resp = turn_batch['resp_gen'])
                turn_batch['客服-生成'] = self.decode_batch(resp_gen)
                spoken, _=self.get_spoken(spoken, turn_batch['客服-生成'], role='system')
        return inverse_transpose_batch(batch)     
        
    def validate_fast(self, data='dev'):
        self.model.eval()
        encoded_data=read_data(self.tokenizer,self.posterior,return_dict=True)
        reserved_keys = ['用户', '查询结果', '查询', '查询-生成', '查询结果-生成'
        , '客服', '客服-生成', '客服-检索', 'bleu', 'inform', ] # reserver keys for demonstration
        if data == 'dev':
            eval_data = encoded_data['dev']
        elif data =='test':
            eval_data = encoded_data['test']
        origin_batch_size=cfg.batch_size
        cfg.batch_size=cfg.eval_batch_size
        batches=get_batches(eval_data,cfg.batch_size)#cfg.batch_size
        result_path=os.path.join(cfg.gpt_path,'result.json') if cfg.gpt else os.path.join(cfg.t5_path,'result.json')
        if cfg.mode == 'train_jsa' or cfg.mode == 'train':
            result_path=os.path.join(cfg.exp_path,'result.json')
        
        if os.path.exists(result_path) and cfg.mode=='test':
            results=json.load(open(result_path, 'r'))
            good_result = []
            new_result = []
            for dial in results:
                new_dial = []
                for tmp_turn in dial:
                    new_turn = {}
                    new_dial.append(new_turn)
                new_result.append(new_dial)
                bleu_score = sum(tmp['bleu'] for tmp in new_turn)/(len(new_turn) + 0.00001)
                if bleu_score>20:
                    good_result.append(new_dial)
            json.dump(good_result, open(result_path.replace('.json', '_good.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
            eval_result=eval_end_to_end(results, retrieve_qa=cfg.retrieve_qa)
            logging.info(eval_result)
            bleu = eval_result['BLEU']
            inform = eval_result['inform']
            bert_score = eval_result['bert_score'].item()
            eval_loss = bleu/50 + inform + bert_score # change eval to matchjsa-krtod paper
            logging.info(eval_loss)
            return eval_result
        else:
            if cfg.retrieve_qa:
                passage_model = SentenceTransformer(cfg.sentence_encoder_path)#' # change to sentence-bert model
                KB_embed, KB_answers = get_kb(passage_model)
                passage_model.eval()
            if cfg.rag_testing:
                if cfg.agent_testing:
                    if cfg.dual_encoder:
                        api_model = SentenceTransformer(cfg.api_encoder_path) # prepare for the cross encoder setting
                        api_embed, api_answers = get_api(api_model, qa=cfg.retrieve_with_qa)
                        api_model.eval()
                    else:
                        api_model = Api_model(cfg, self.tokenizer)#,tokenizer
                        api_model.load_state_dict(torch.load(os.path.join(cfg.api_save_dir, "model.pt")))
                        api_model.to(cfg.device[0])#is list
                        api_embed = None
                        api_answers = label_api_dict
                        api_model.eval()
                else:
                    passage_model = SentenceTransformer(cfg.sentence_encoder_path)
                    passage_model.eval()
                    global_index = build_global_index(passage_model, self.tokenizer, batch_size=16)

                    context_model = Retrieval_model(cfg, self.tokenizer)
                    context_model.load_state_dict(torch.load(os.path.join(cfg.apiret_save_dir, "model.pt")))
                    context_model.to(cfg.device[0])

            result_collection = []
            st=time.time()
            for batch in batches:
                try:
                    if batch==[]:
                        continue
                    if cfg.rag_training:
                        if cfg.rag_testing:
                            batch=self.generate_batch_turn_level(batch, ground_truth_db=cfg.gt_db, passage_model=passage_model, KB_embed=global_index,
                            api_model=context_model)
                        else:
                            batch=self.generate_batch_turn_level(batch, ground_truth_db=cfg.gt_db)
                    elif cfg.only_response:
                        batch=self.generate_batch_turn_level(batch, ground_truth_db=cfg.gt_db,
                        passage_model=passage_model, KB_embed=KB_embed, KB_answers=KB_answers, 
                        api_model=api_model, api_embed=api_embed, api_answers=api_answers)
                    elif cfg.no_retrieval:
                        batch=self.generate_batch_turn_level(batch, ground_truth_db=cfg.gt_db)
                    else:
                        batch=self.generate_batch_turn_level(batch, ground_truth_db=cfg.gt_db,
                        passage_model=passage_model, KB_embed=KB_embed, KB_answers=KB_answers)
                    for dialog in batch: 
                        result_collection.append(inverse_transpose_turn(dialog))
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        logging.info("WARNING: ran out of memory during validation and batch will be divided by half, batch size:{}, turn num:{}"\
                            .format(len(batch), len(batch[0])))
                        if hasattr(torch.cuda, 'empty_cache'):
                            with torch.cuda.device(self.device):
                                torch.cuda.empty_cache()
                        #divide the batch in half if out of memory
                        batches.insert(0,batch[:len(batch)//2])
                        batches.insert(1,batch[len(batch)//2:])
                    else:
                        logging.info(str(exception))
                        raise exception
            logging.info('Inference time:{:.3f} min'.format((time.time()-st)/60))
            eval_result, tmp_result = eval_end_to_end(result_collection, return_results=True, retrieve_qa=cfg.retrieve_qa)
            json.dump(tmp_result, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

            if cfg.mode=='test':
                new_result = []
                good_result = []
                for dial in tmp_result:
                    new_dial = []
                    for tmp_turn in dial:
                        new_turn = {}
                        for k in reserved_keys:
                            if k in tmp_turn:
                                new_turn[k] = tmp_turn[k]
                        new_dial.append(new_turn)
                    new_result.append(new_dial)
                    bleu_score = sum(tmp['bleu'] for tmp in new_dial)/len(new_turn)
                    if bleu_score>20:
                        good_result.append(new_dial)
                #json.dump(new_result, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
                json.dump(good_result, open(result_path.replace('.json', '_good.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
            logging.info(eval_result)
            cfg.batch_size = origin_batch_size
            return eval_result

    def validate_post(self, data='dev'):
        if cfg.mode == 'test_post':
            result_path=os.path.join(cfg.posterior_path,'result.json')
        else:
            result_path=os.path.join(cfg.exp_path,'result.json')
            if data == 'train':
                result_path=os.path.join(cfg.exp_path,'result_train.json')
        if os.path.exists(result_path) and cfg.mode=='test_post':
            results = json.load(open(result_path, 'r'))
            eval_result=eval_post(results)
            logging.info(eval_result)
            ui = eval_result['P/R/F1 for user intent'][2]
            si = eval_result['P/R/F1 for system intent'][2]
            db = eval_result['P/R/F1 for db prediction'][2] 
            eval_loss = ui + si + db
            logging.info(eval_loss)
            return eval_result
        
        else:
            self.model.eval()
            encoded_data=read_data(self.tokenizer,self.posterior,return_dict=True)
            if data == 'dev':
                eval_data = encoded_data['dev']
            if data == 'train':
                eval_data = encoded_data['train'][:1000]
            elif data =='test':
                eval_data = encoded_data['test']
            origin_batch_size=cfg.batch_size
            cfg.batch_size=cfg.eval_batch_size
            batches=get_batches(eval_data,cfg.batch_size)#cfg.batch_size
            result_collection = []
            st=time.time()
            for batch_idx, batch in enumerate(batches):
                pv_batch=None
                dial_batch=transpose_batch(batch)
                try:
                    for turn_num, turn_batch in enumerate(dial_batch):
                        turn_batch, pv_batch==self.gen_hidden_state(turn_batch, pv_batch,validate=True)
                    dial_batch=inverse_transpose_batch(dial_batch)
                    for dialog in dial_batch:
                        result_collection.append(inverse_transpose_turn(dialog))
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        logging.info("WARNING: ran out of memory during validation and batch will be divided by half, batch size:{}, turn num:{}"\
                            .format(len(batch),len(batch[0])))
                        if hasattr(torch.cuda, 'empty_cache'):
                            with torch.cuda.device(self.device):
                                torch.cuda.empty_cache()
                        #divide the batch in half if out of memory
                        batches.insert(0,batch[:len(batch)//2])
                        batches.insert(1,batch[len(batch)//2:])
                    else:
                        logging.info(str(exception))
                        raise exception
            logging.info('Inference time:{:.3f} min'.format((time.time()-st)/60))
            json.dump(result_collection, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
            eval_result=eval_post(result_collection)
            logging.info(eval_result)
            cfg.batch_size = origin_batch_size
            return eval_result

    def generate_pseudo_label(self):
        result_path='data/encoded_pseudo_data_new.json'
        encoded_file = os.path.join(cfg.data_dir, 'encoded_data_unl_whole.json')
        unl=json.load(open(encoded_file, 'r', encoding='utf-8'))
        #dials_unl=get_unsup(self.tokenizer)

        self.model.eval()
        origin_batch_size=cfg.batch_size
        cfg.batch_size=cfg.eval_batch_size
        batches=get_batches(unl,cfg.batch_size)#cfg.batch_size
        result_collection = []
        st=time.time()
        for batch in tqdm(batches):
            pv_batch=None
            dial_batch=transpose_batch(batch)
            try:
                for turn_num, turn_batch in enumerate(dial_batch):
                    turn_batch, pv_batch==self.gen_hidden_state(turn_batch, pv_batch)
                dial_batch=inverse_transpose_batch(dial_batch)
                for dialog in dial_batch:
                    result_collection.append(inverse_transpose_turn(dialog))
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logging.info("WARNING: ran out of memory during validation and batch will be divided by half, batch size:{}, turn num:{}"\
                        .format(len(batch),len(batch[0])))
                    if hasattr(torch.cuda, 'empty_cache'):
                        with torch.cuda.device(self.device):
                            torch.cuda.empty_cache()
                    #divide the batch in half if out of memory
                    batches.insert(0,batch[:len(batch)//2])
                    batches.insert(1,batch[len(batch)//2:])
                else:
                    logging.info(str(exception))
                    raise exception
        logging.info('Inference time:{:.3f} min'.format((time.time()-st)/60))
        for dial in result_collection:
            for turn in dial:
                turn.pop('user_decode')
                turn.pop('resp_decode')
                turn.pop('db_decode')
                turn.pop('db_new')
                turn.pop('db')
        json.dump(result_collection, open(result_path, 'w', encoding='utf-8'), indent=2)
        return

    def generate_batch(self, model, contexts, max_len, eos_id, beam=1, return_prob=False):
        # generate by batch
        # contexts: a list of ids
        # max_len: the max generated length
        # eos_id: the end id
        # return: a batch of ids with pre pad 
        batch_size=len(contexts)
        # log prob of the generated text
        probs = [[0.0 for i in range(beam)] for j in range(batch_size)]
        end_flag=np.zeros(batch_size)
        if beam>1:
            beam_box=[beam]*batch_size
            beam_result=[[] for _ in range(batch_size)]
            max_prob=[-float('inf')]*batch_size
        past_key_values=None
        inputs,attentions=batch_align(contexts,left_len=max_len,return_attn=True)
        inputs=torch.tensor(inputs).to(model.device)
        attentions=torch.tensor(attentions).to(model.device)
        model.eval()
        with torch.no_grad():
            for i in range(max_len):
                if beam==1:
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    if inputs.size(0)==0:
                        raise ValueError(contexts, inputs.cpu().list(), attentions)
                    outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)

                    past_key_values=outputs.past_key_values

                    preds=outputs.logits[:,-1,:].argmax(-1)#B

                    # calculating probs
                    case_probs = F.softmax(outputs.logits[:,-1,:]/cfg.temperature,dim=-1)
                    for num in range(preds.shape[0]):
                        index = preds.cpu().tolist()[num]
                        probs[num][0] = probs[num][0] + math.log(case_probs[num,index].cpu().item())

                    if i==0:
                        gen_tensor=preds.unsqueeze(1)
                    else:
                        gen_tensor=torch.cat([gen_tensor,preds.unsqueeze(1)], dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                else:
                    if i==0:
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values)
                        past_key_values=[outputs.past_key_values]*beam
                        log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                        beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                        gen_tensor=beam_idx.unsqueeze(-1)# B, beam, 1
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                        pv_beam_prob=beam_prob #B, beam
                        pv_beam_idx=beam_idx#B, beam
                    else:
                        for j in range(beam):
                            inputs=pv_beam_idx[:,j].unsqueeze(-1) # B, 1
                            outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values[j])
                            past_key_values[j]=outputs.past_key_values
                            log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                            beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                            if j==0:
                                prob_pool= beam_prob + pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam) # B, beam
                                id_pool=beam_idx
                            else:
                                prob_pool=torch.cat([prob_pool, beam_prob+pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam)], -1) # B, beam*beam
                                id_pool=torch.cat([id_pool, beam_idx], -1)# B, beam*beam
                        beam_prob, temp_id=torch.topk(prob_pool, beam, -1) #B, beam
                        beam_idx=torch.gather(id_pool, -1, temp_id)
                        temp_id=temp_id//beam
                        # past_key_values is a list of length beam, which
                        #past_key_values (Tuple[Tuple[torch.Tensor]] — Tuple of length config.n_layers, containing tuples of tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)).

                        new_past_key_values=copy.deepcopy(past_key_values)
                        for b in range(batch_size):
                            gen_tensor[b, :, :]=gen_tensor[b, :, :].index_select(0, temp_id[b, :])
                            for t in range(beam):
                                for l in range(model.config.n_layer):
                                    for qk in range(2):
                                        new_past_key_values[t][l][qk][b, :, :, :]=past_key_values[temp_id[b, t]][l][qk][b, :, :, :]
                        past_key_values=new_past_key_values
                        #past_key_values=[past_key_values[t] for t in temp_id.cpu().list()]
                        gen_tensor=torch.cat([gen_tensor, beam_idx.unsqueeze(-1)],-1) #B, beam, T
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                        pv_beam_prob=beam_prob #B, beam
                        pv_beam_idx=beam_idx
                    for m in range(batch_size):
                        for n, gen in enumerate(gen_tensor.cpu().tolist()[m]):
                            if (eos_id in gen or i==max_len-1) and len(beam_result[m])<beam:
                                #beam_box[m]-=1
                                avg_prob=copy.deepcopy(pv_beam_prob[m][n].item())
                                beam_result[m].append((gen, avg_prob))
                                pv_beam_prob[m][n]=-float('inf')
                    if all([len(tmp_result)==beam for tmp_result in beam_result]):
                        break  
                    #probs = beam_prob.cpu().item()

        if beam==1:
            if return_prob:
                return gen_tensor.cpu().tolist(), probs
            else:
                return gen_tensor.cpu().tolist()
        else:
            """
            for m in range(batch_size):
                for n, gen in enumerate(gen_tensor.cpu().tolist()[m]):
                    if eos_id in gen:
                        beam_box[m]-=1
                        # focus on the prob
                        avg_prob=pv_beam_prob[m][n].item() #len(gen)
                        beam_result[m].append((gen, avg_prob))
            """
                        #pv_beam_prob[m][n]=-float('inf')
            # we do not break during beam search
            #if not any(beam_box):
                #   break
            #for i, tup in enumerate(beam_result):
            #    beam_list=sorted(tup, key=lambda item:item[1], reverse=True)
            #    beam_result[i]=[item[0] for item in beam_list[:beam]]
            return beam_result

    def convert_batch_ids_to_tokens(self, tokenizer, input_ids, eos_id, return_ids=False):
        # input_ids: B*T
        # output: B*string
        outputs=[]
        outputs_ids=[]
        for sent_ids in input_ids:
            if eos_id in sent_ids:
                sent_ids=sent_ids[:sent_ids.index(eos_id)+1]
            else:
                sent_ids[-1]=eos_id
            outputs_ids.append(sent_ids)
            outputs.append(tokenizer.decode(sent_ids))
        if return_ids:
            return outputs, outputs_ids
        return outputs


def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return cfg
    
class Semi_supervision(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device=cfg.device
        self.tokenizer = BertTokenizer.from_pretrained(cfg.posterior_path) # cfg.gpt_path
        self.cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep = self.tokenizer.convert_tokens_to_ids('[SEP]')
        if isinstance(self.device,list):
            self.device1 = self.device[0]
            self.device2 = self.device[1]
            self.device3 = self.device[-1]
            self.PrioriModel=Model(self.device1)#GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
            self.PosteriorModel=Model(self.device2,posterior=True)

        self.bert_tokenizer = BertTokenizer.from_pretrained(cfg.bert_save_path)
        if cfg.only_one_model:
            self.bert_model = BertForNextSentencePrediction.from_pretrained(cfg.bert_save_path)#EncoderModel(cfg,tokenizer)
            self.bert_model.to(self.device2)
        else:
            self.bert_model = EncoderModel(cfg,self.bert_tokenizer)
        # load ebm
        if cfg.jsa_ebm:            
            cfg.add_extra_feature = False
            # ebm
            cfg.use_all_proposal = True
            # mis settings
            cfg.train_ebm_mis = False # modified, MIS or not
            if cfg.train_ebm_mis:
                cfg.use_all_proposal = True # mis uses all proposals  
            cfg.residual = False # residual or not

            self.ebm = EBM(cfg, self.bert_tokenizer)
            if cfg.ebm_save_path == 'ret_exp/best_ebm':
                self.ebm_save_path = os.path.join(cfg.ebm_save_path, 
                f"model_allproposal{cfg.use_all_proposal}_mis_cache{cfg.train_ebm_mis}_residual{cfg.residual}"+ ("_add_feature" if cfg.add_extra_feature else "") + ".pt")
            else:
                self.ebm_save_path = cfg.ebm_save_path
            self.ebm.load_state_dict(torch.load(self.ebm_save_path, map_location=torch.device('cpu')))
            self.ebm.to(self.device3)
        self.bert_model.eval()

        json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
    
        logging.info('Special token added, vocab size:{}'.format( len(self.tokenizer)))

        # log
        log_path='./log/log_{}'.format(cfg.exp_name)
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
            os.mkdir(log_path)
        else:
            os.mkdir(log_path)
        self.tb_writer = SummaryWriter(log_dir=log_path)
        #track nan in ebm training
        self.nan_num = 0 

    def jsa_train(self):
        cfg = self.cfg
        logging.info('------Running joint stochastic approximation------')
        # unlab_ratio = 1 # hyperparameter for num of jsa-training and supervised-training, type int, not use change the function 
        # in reader.get_unsup instead
        # use supervised sample multiple times if unlab_ratio is high, for example 4 and 9
        SUP_AUG = 3
        cfg.batch_size=cfg.batch_size*cfg.gradient_accumulation_steps #batch_size changed
        
        #dev_dataloader=DataLoader(encoded_data['dev'], batch_size=cfg.eval_batch_size, collate_fn=train_collate_fn)
        encoded_data=read_data(self.tokenizer,return_dict=True)
        train_data=encoded_data['train']
        random.shuffle(train_data)
        batches_lab = get_batches(train_data,cfg.batch_size)
        #num_dials=len(train_data)

        dials_unl=get_unsup(self.tokenizer,len(train_data))#list of unlabeled dialogs
        
        batches_unl = get_batches(dials_unl , batch_size = cfg.batch_size)
        logging.info('Labeled dials:{}, Unlabeled dials:{}'.format(len(train_data),len(dials_unl)))
        turn_nums = sum(len(dial) for dial in train_data) + sum(len(dial1) for dial1 in dials_unl)

        all_batches = []
        if cfg.debugging:
            batches_unl=batches_unl[9*len(batches_unl)//10:]
        for i in range(SUP_AUG):
            for batch in batches_lab:
                all_batches.append({'batch':transpose_batch(batch),'supervised':True})
        for batch in batches_unl:
            all_batches.append({'batch':transpose_batch(batch),'supervised':False})

        optimizer1, scheduler1 = self.PrioriModel.get_optimizers(turn_nums, self.PrioriModel.model) #num of total turns
        optimizer2, scheduler2 = self.PosteriorModel.get_optimizers(turn_nums, self.PosteriorModel.model)

        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info('  Num Batches = %d', len(all_batches))

        global_step = 0
        max_score=0
        min_loss=10000
        log_inputs = 3

        for epoch in range(cfg.epoch_num):
            GENERATE = True # if (epoch%2 == 0) else False # 2 can be modified to a larger number
            epoch_step = 0
            tr_loss, sup_loss, uns_loss = 0.0, 0.0, 0.0
            sup_step, uns_step=0, 0
            btm = time.time()
            self.PrioriModel.model.zero_grad()
            self.PosteriorModel.model.zero_grad()
            random.shuffle(all_batches)
            for batch_idx, dial_batch_dict in tqdm(enumerate(all_batches)):
                #unsup_train
                pv_batch=None
                spoken = []
                hist = []
                for turn_num, turn_batch in enumerate(dial_batch_dict['batch']):
                    if (not dial_batch_dict['supervised']) and GENERATE:
                        turn_batch, next_pv_batch=self.PosteriorModel.gen_hidden_state(turn_batch, pv_batch)
                    else:
                        next_pv_batch=self.PosteriorModel.get_pv_batch(pv_batch, user=turn_batch['user'], resp=turn_batch['resp'], entity=turn_batch['entity'])
                    first_turn = (turn_num == 0)
                    mini_batches, mini_pv_batches=split_turn_batch(turn_batch, cfg.origin_batch_size, other_batch=pv_batch)
                    if (not dial_batch_dict['supervised']) and GENERATE:
                        for i, batch in enumerate(mini_batches):
                            mini_pv_batch=None if turn_num==0 else mini_pv_batches[i]
                            inputs_prior, labels_prior = convert_batch_turn(config=cfg, cls=self.cls,
                            sep=self.sep, turn_batch=batch, pv_batch=mini_pv_batch, 
                            first_turn=first_turn, posterior=False)
                            inputs_posterior, labels_posterior = convert_batch_turn(config=cfg, cls=self.cls,
                            sep=self.sep, turn_batch=batch, pv_batch=mini_pv_batch,
                            first_turn=first_turn, posterior=False)
                            self.PrioriModel.model.train()
                            self.PosteriorModel.model.train()

                            if len(spoken) < (i+1):
                                spoken.append([])
                                hist.append([])
                            spoken[i], hist[i]=self.get_spoken(spoken[i], batch['user_decode'], role='system')

                            if log_inputs > 0 :  # log inputs for the very first two turns
                                tmp_prior = self.tokenizer.decode(inputs_prior['contexts'][0])
                                tmp_posterior = self.tokenizer.decode(inputs_posterior['contexts'][0])
                                logging.info('Prior examples:\n{}'.format(tmp_prior))
                                logging.info("Posterior examples:\n{}".format(tmp_posterior))
                                log_inputs -= 1

                            jsa_labels=(copy.deepcopy(inputs_posterior),copy.deepcopy(labels_posterior),copy.deepcopy(inputs_prior),copy.deepcopy(labels_prior),copy.deepcopy(batch['db_decode']))
                            # to tensor
                            inputs_prior = self.add_torch_input(inputs_prior)#B,T
                            inputs_posterior = self.add_torch_input(inputs_posterior,posterior=True)
                            labels_prior=self.add_torch_input(labels_prior)#B,T
                            labels_posterior=self.add_torch_input(labels_posterior,posterior=True)
                            # loss
                            with torch.no_grad():
                                outputs_prior=self.PrioriModel.model(inputs_prior['contexts_tensor'])
                                outputs_posterior=self.PosteriorModel.model(inputs_posterior['contexts_tensor'])#B,T,V
                                logits_pri=outputs_prior[0]
                                logits_post=outputs_posterior[0]
                            
                            #get prob
                            jsa_prob=self.get_jsa_prob(logits_pri,logits_post,\
                                    labels_prior['contexts_tensor'],labels_posterior['contexts_tensor'])
                            retrieval_prob = self.get_retrieval_prob(hist[i], batch['db_decode'])
                            for jsa_count in range(len(jsa_prob)):
                                jsa_prob[jsa_count] = jsa_prob[jsa_count] + retrieval_prob[jsa_count] 
                            if epoch==0:
                                last_prob=jsa_prob #accept the proposal at the first turn
                            else:
                                
                                t_label=turn_batch['jsa_labels'][i]
                                temp_label=copy.deepcopy(t_label)
                                i_posterior=self.add_torch_input(temp_label[0],posterior=True)
                                l_posterior=self.add_torch_input(temp_label[1],posterior=True)
                                i_prior=self.add_torch_input(temp_label[2])
                                l_prior=self.add_torch_input(temp_label[3])
                                with torch.no_grad():
                                    o_prior=self.PrioriModel.model(i_prior['contexts_tensor'])
                                    o_posterior=self.PosteriorModel.model(i_posterior['contexts_tensor'])#B,T,V
                                    lo_pri=o_prior[0]
                                    lo_post=o_posterior[0]
                                
                                last_prob=self.get_jsa_prob(lo_pri,lo_post,\
                                        l_prior['contexts_tensor'],l_posterior['contexts_tensor'])
                                l_retrieval_prob = self.get_retrieval_prob(hist[i], temp_label[4])
                                for jsa_count in range(len(last_prob)):
                                    last_prob[jsa_count] = last_prob[jsa_count] + l_retrieval_prob[jsa_count]

                            spoken[i], _=self.get_spoken(spoken[i], batch['resp_decode'], role='system') # need to be modified
                            for prob_num in range(min(len(jsa_prob),len(last_prob))):
                                if jsa_prob[prob_num]-last_prob[prob_num]>0:
                                    ratio=1.0
                                else:
                                    ratio=math.exp(jsa_prob[prob_num]-last_prob[prob_num])
                                if ratio<1.0:
                                    if random.random()>ratio:
                                        for j in range(5):
                                            if 'contexts_np' in jsa_labels[j]:
                                                jsa_labels[j].pop('contexts_np')
                                            if j!=4:
                                                jsa_labels[j]['contexts'][prob_num]=turn_batch['jsa_labels'][i][j]['contexts'][prob_num]
                                                jsa_labels[j]['lengths'][prob_num]=turn_batch['jsa_labels'][i][j]['lengths'][prob_num]
                                            else:
                                                jsa_labels[j][prob_num] = turn_batch['jsa_labels'][i][j][prob_num]
                            if epoch==0:
                                if 'jsa_labels' not in turn_batch:
                                    turn_batch['jsa_labels']=[]
                                turn_batch['jsa_labels'].append(jsa_labels)
                            else:
                                turn_batch['jsa_labels'][i]=jsa_labels
                            temp_label=copy.deepcopy(jsa_labels)
                            inputs_posterior=self.add_torch_input(temp_label[0],posterior=True)
                            labels_posterior=self.add_torch_input(temp_label[1],posterior=True)
                            inputs_prior=self.add_torch_input(temp_label[2])
                            labels_prior=self.add_torch_input(temp_label[3])
                            if epoch==0:
                                pass
                            else:
                                outputs1=self.PrioriModel.model(inputs_prior['contexts_tensor'])
                                loss_pri=self.PrioriModel.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                                outputs2=self.PosteriorModel.model(inputs_posterior['contexts_tensor'])
                                loss_pos=self.PosteriorModel.calculate_loss_and_accuracy(outputs2,labels_posterior['contexts_tensor'])
                            
                            if epoch!=0:
                                loss_pri.backward()
                                loss_pos.backward()
                                loss=loss_pri.item()+loss_pos.item()#.to(self.device1), .to(self.device2)
                                tr_loss += loss
                                uns_loss += loss
                                uns_step+=1
                    else:
                        if epoch!=0:
                            for i, batch in enumerate(mini_batches):
                                if dial_batch_dict['supervised']:
                                    mini_pv_batch=None if turn_num==0 else mini_pv_batches[i]
                                    inputs_prior, labels_prior = convert_batch_turn(config=cfg, cls=self.cls, 
                                    sep=self.sep, turn_batch=batch, pv_batch=mini_pv_batch,
                                    first_turn=first_turn, posterior=False)
                                    inputs_posterior, labels_posterior = convert_batch_turn(cconfig=cfg, cls=self.cls, 
                                    sep=self.sep, turn_batch=batch, pv_batch=mini_pv_batch, 
                                    first_turn=first_turn, posterior=True)
                                else:
                                    jsa_labels = copy.deepcopy(turn_batch['jsa_labels'][i])
                                    inputs_posterior = jsa_labels[0]
                                    labels_posterior = jsa_labels[1]
                                    inputs_prior = jsa_labels[2]
                                    labels_prior = jsa_labels[3]
                                inputs_prior = self.add_torch_input(inputs_prior)#B,T
                                labels_prior=self.add_torch_input(labels_prior)#B,T
                                inputs_posterior=self.add_torch_input(inputs_posterior,posterior=True)
                                labels_posterior=self.add_torch_input(labels_posterior,posterior=True)

                                outputs1 = self.PrioriModel.model(inputs_prior['contexts_tensor'])
                                loss_pri=self.PrioriModel.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                                loss_pri.backward()
                                if dial_batch_dict['supervised']:
                                    outputs2=self.PosteriorModel.model(inputs_posterior['contexts_tensor'])
                                    loss_pos=self.PosteriorModel.calculate_loss_and_accuracy(outputs2,labels_posterior['contexts_tensor'])
                                    loss_pos.backward()

                                loss = loss_pri.item() + (loss_pos.item() if dial_batch_dict['supervised'] else 0)
                                tr_loss += loss
                                sup_loss +=loss
                                sup_step +=1
                    if epoch!=0:
                        torch.nn.utils.clip_grad_norm_(self.PrioriModel.model.parameters(), 5.0)
                        torch.nn.utils.clip_grad_norm_(self.PosteriorModel.model.parameters(), 5.0)
                        epoch_step+=1
                        optimizer1.step()
                        optimizer1.zero_grad()
                        optimizer2.step()
                        optimizer2.zero_grad()
                        global_step+=1
                        #if cfg.use_scheduler:
                        scheduler1.step()
                        scheduler2.step()
                        if self.tb_writer:
                            self.tb_writer.add_scalar('lr1', optimizer1.param_groups[0]["lr"],global_step)
                            self.tb_writer.add_scalar('lr2', optimizer2.param_groups[0]["lr"],global_step)
                            self.tb_writer.add_scalar('loss', loss, global_step)
                    pv_batch=next_pv_batch
                    
            logging.info('Epoch: {}, Train epoch time: {:.2f} min, loss:{:.3f}, avg_sup_loss:{:.3f}, avg_uns_loss:{:.3f}'.format(epoch, 
                (time.time()-btm)/60, tr_loss/(epoch_step+1e-10), sup_loss/(sup_step+1e-10), uns_loss/(uns_step+1e-10)))

            if cfg.save_type =='max_score' and (epoch!=0):
                cfg.gt_db = False
                cfg.retrieve_kb = True
                eval_result=self.PrioriModel.validate_fast() # 'test'
                cfg.retrieve_kb = False
                cfg.gt_db = True
                bleu = eval_result['BLEU']
                f1_query = eval_result['query_f1'][2]
                qa = eval_result['qa'][2]
                logging.info('user:{:.3f}, system:{:.3f} , bleu:{:.3f}, success:{:.3f}'.format(ui, si, bleu, success))
                eval_loss = bleu/25 + qa*2 + f1_query
                logging.info('Epoch:{}, Train epoch time:{:.2f} min, epoch loss:{:.3f}, eval loss:{:.3f}'.format(epoch, (time.time()-btm)/60, tr_loss, eval_loss))
                self.tb_writer.add_scalar('eval_loss', eval_loss, epoch)
                if max_score < eval_loss:
                    max_score=eval_loss
                    self.PrioriModel.save_model()
                    self.PosteriorModel.save_model()
                else:
                    self.PrioriModel.save_model('last_model')
                    #self.PosteriorModel.save_model('last_model')
            elif cfg.save_type =='min_loss':
                if eval_loss<min_loss:
                    min_loss=eval_loss
                    self.PrioriModel.save_model()
                    self.PosteriorModel.save_model()

    # joint training energy-based model with generation model
    def jsa_train_ebm(self):
        cfg = self.cfg
        logging.info('------Running joint stochastic approximation with energy based model------')
        # unlab_ratio = 1 # hyperparameter for num of jsa-training and supervised-training, type int, not use change the function 
        # in reader.get_unsup instead
        # use supervised sample multiple times if unlab_ratio is high, for example 4 and 9
        cfg.batch_size=cfg.batch_size*cfg.gradient_accumulation_steps #batch_size changed
        #encoded_data=read_data(self.tokenizer)
        #encoded_data_post=read_data(self.tokenizer,posterior=True)
        #train_data = encoded_data['train']
        #train_data_post = encoded_data_post['train']
        
        #dev_dataloader=DataLoader(encoded_data['dev'], batch_size=cfg.eval_batch_size, collate_fn=train_collate_fn)
        encoded_data=read_data(self.tokenizer,return_dict=True)
        train_data=encoded_data['train']
        random.shuffle(train_data)
        batches_lab = get_batches(train_data,cfg.batch_size)
        #num_dials=len(train_data)

        dials_unl=get_unsup(self.tokenizer,len(train_data))#list of unlabeled dialogs
        
        batches_unl = get_batches(dials_unl , batch_size = cfg.batch_size)
        logging.info('Labeled dials:{}, Unlabeled dials:{}'.format(len(train_data),len(dials_unl)))
        turn_nums = sum(len(dial) for dial in train_data) + sum(len(dial1) for dial1 in dials_unl)

        all_batches = []
        
        SUP_AUG = 3 # control the num of sup-aug num

        #only aug sup data in first 4 epochs
        if cfg.jsa_unsup<4:
            EPOCH_TH = 3
        elif cfg.jsa_unsup==4:
            EPOCH_TH = 4 
        elif cfg.jsa_unsup==9:
            EPOCH_TH = 20
        #EBM_EPOCH_TH = 1 # train ebm after the first epochs
        if cfg.debugging:
            batches_unl=batches_unl[19*len(batches_unl)//20:]
            batches_lab=batches_lab[19*len(batches_lab)//20:]
        for i in range(SUP_AUG):
            for batch in batches_lab:
                all_batches.append({'batch':transpose_batch(batch),'supervised':True})
        for batch in batches_unl:
            all_batches.append({'batch':transpose_batch(batch),'supervised':False})

        optimizer1, scheduler1 = self.PrioriModel.get_optimizers(turn_nums, self.PrioriModel.model) #num of total turns
        optimizer2, scheduler2 = self.PosteriorModel.get_optimizers(turn_nums, self.PosteriorModel.model)
        #if cfg.jsa_ebm:
        # make the lr of ebm smaller to avoid nan
        optimizer3, scheduler3 = get_optimizers(num_samples=turn_nums ,model=self.ebm, lr=cfg.lr/10)

        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info('  Num Batches = %d', len(all_batches))

        global_step = 0
        max_score=0
        min_loss=10000
        log_inputs = 3

        for epoch in range(cfg.epoch_num):
            GENERATE = True # if (epoch%2 == 0) else False # 2 can be modified to a larger number
            if epoch>EPOCH_TH: # reset all_batches to reduce sup-num
                for batch in all_batches:
                    if batch['supervised']==True:
                        all_batches.remove(batch)
                for batch in batches_lab:
                    all_batches.append({'batch':transpose_batch(batch),'supervised':True})
            epoch_step = 0
            tr_loss, sup_loss, uns_loss, ebm_loss_label, ebm_loss_unlabel  = 0.0, 0.0, 0.0, 0.0, 0.0
            sup_step, uns_step=0, 0
            btm = time.time()
            self.PrioriModel.model.zero_grad()
            self.PosteriorModel.model.zero_grad()
            random.shuffle(all_batches)
            for batch_idx, dial_batch_dict in tqdm(enumerate(all_batches)):
                #unsup_train
                pv_batch=None
                spoken = []
                hist = []
                for turn_num, turn_batch in enumerate(dial_batch_dict['batch']):
                    if (not dial_batch_dict['supervised']) and GENERATE:
                        turn_batch, next_pv_batch=self.PosteriorModel.gen_hidden_state(turn_batch, pv_batch)
                    else:
                        next_pv_batch=self.PosteriorModel.get_pv_batch(pv_batch, user=turn_batch['user'], resp=turn_batch['resp'], entity=turn_batch['entity'])
                    first_turn = (turn_num == 0)
                    mini_batches, mini_pv_batches=split_turn_batch(turn_batch, cfg.origin_batch_size, other_batch=pv_batch)
                    if (not dial_batch_dict['supervised']) and GENERATE:
                        for i, batch in enumerate(mini_batches):
                            mini_pv_batch=None if turn_num==0 else mini_pv_batches[i]
                            inputs_prior, labels_prior = convert_batch_turn(config=cfg, cls=self.cls,
                            sep=self.sep, turn_batch=batch, pv_batch=mini_pv_batch, 
                            first_turn=first_turn, posterior=False)
                            inputs_posterior, labels_posterior = convert_batch_turn(config=cfg, cls=self.cls, 
                            sep=self.sep, turn_batch=batch, pv_batch=mini_pv_batch, 
                            first_turn=first_turn, posterior=True)
                            self.PrioriModel.model.train()
                            self.PosteriorModel.model.train()

                            if len(spoken) < (i+1):
                                spoken.append([])
                                hist.append([])
                            spoken[i], hist[i]=self.get_spoken(spoken[i], batch['user_decode'], role='user')

                            if log_inputs > 0 :  # log inputs for the very first two turns
                                tmp_prior = self.tokenizer.decode(inputs_prior['contexts'][0])
                                tmp_posterior = self.tokenizer.decode(inputs_posterior['contexts'][0])
                                logging.info('Prior examples:\n{}'.format(tmp_prior))
                                logging.info("Posterior examples:\n{}".format(tmp_posterior))
                                #print(tmp_prior)
                                #print(tmp_posterior)
                                log_inputs -= 1

                            jsa_labels=(copy.deepcopy(inputs_posterior),copy.deepcopy(labels_posterior),copy.deepcopy(inputs_prior),copy.deepcopy(labels_prior),copy.deepcopy(batch['db_decode']))
                            # to tensor
                            inputs_prior = self.add_torch_input(inputs_prior)#B,T
                            inputs_posterior = self.add_torch_input(inputs_posterior,posterior=True)
                            labels_prior=self.add_torch_input(labels_prior)#B,T
                            labels_posterior=self.add_torch_input(labels_posterior,posterior=True)
                            # loss
                            with torch.no_grad():
                                outputs_prior=self.PrioriModel.model(inputs_prior['contexts_tensor'])
                                outputs_posterior=self.PosteriorModel.model(inputs_posterior['contexts_tensor'])#B,T,V
                                logits_pri=outputs_prior[0]
                                logits_post=outputs_posterior[0]
                            
                            #get prob
                            jsa_prob=self.get_jsa_prob(logits_pri,logits_post,\
                                    labels_prior['contexts_tensor'],labels_posterior['contexts_tensor'])
                            retrieval_prob = self.get_retrieval_prob_ebm(hist[i], batch['db_decode'])
                            for jsa_count in range(len(jsa_prob)):
                                jsa_prob[jsa_count] = jsa_prob[jsa_count] + retrieval_prob[jsa_count] 
                            if epoch==0:
                            #if epoch>-1:
                                last_prob=jsa_prob #accept the proposal at the first turn
                                #if 'prob' not in turn_batch:
                                #    turn_batch['prob']=[]
                                #turn_batch['prob'].append(jsa_prob)
                            else:
                                
                                t_label=turn_batch['jsa_labels'][i]
                                temp_label=copy.deepcopy(t_label)
                                i_posterior=self.add_torch_input(temp_label[0],posterior=True)
                                l_posterior=self.add_torch_input(temp_label[1],posterior=True)
                                i_prior=self.add_torch_input(temp_label[2])
                                l_prior=self.add_torch_input(temp_label[3])
                                with torch.no_grad():
                                    o_prior=self.PrioriModel.model(i_prior['contexts_tensor'])
                                    o_posterior=self.PosteriorModel.model(i_posterior['contexts_tensor'])#B,T,V
                                    lo_pri=o_prior[0]
                                    lo_post=o_posterior[0]
                                
                                #get prob
                                last_prob=self.get_jsa_prob(lo_pri,lo_post,\
                                        l_prior['contexts_tensor'],l_posterior['contexts_tensor'])
                                l_retrieval_prob = self.get_retrieval_prob_ebm(hist[i], temp_label[4])
                                for jsa_count in range(len(last_prob)):
                                    last_prob[jsa_count] = last_prob[jsa_count] + l_retrieval_prob[jsa_count]
                                #last_prob=copy.deepcopy(turn_batch['prob'][i])
                                #turn_batch['prob'][i]=jsa_prob
                                #last_prob=jsa_prob
                            
                            #update bspn
                            for prob_num in range(min(len(jsa_prob),len(last_prob))):
                                if jsa_prob[prob_num]-last_prob[prob_num]>0:
                                    ratio=1.0
                                else:
                                    ratio=math.exp(jsa_prob[prob_num]-last_prob[prob_num])
                                if ratio<1.0:
                                    if random.random()>ratio:
                                        for j in range(5):
                                            if 'contexts_np' in jsa_labels[j]:
                                                jsa_labels[j].pop('contexts_np')
                                            if j!=4:
                                                jsa_labels[j]['contexts'][prob_num]=turn_batch['jsa_labels'][i][j]['contexts'][prob_num]
                                                #jsa_labels[j]['contexts_np'][prob_num]=dial_batch_dict['jsa_labels'][j]['contexts_np'][prob_num]
                                                jsa_labels[j]['lengths'][prob_num]=turn_batch['jsa_labels'][i][j]['lengths'][prob_num]
                                            else:
                                                jsa_labels[j][prob_num] = turn_batch['jsa_labels'][i][j][prob_num]
                            if epoch==0:
                                if 'jsa_labels' not in turn_batch:
                                    turn_batch['jsa_labels']=[]
                                turn_batch['jsa_labels'].append(jsa_labels)
                            else:
                                turn_batch['jsa_labels'][i]=jsa_labels
                            temp_label=copy.deepcopy(jsa_labels)
                            inputs_posterior=self.add_torch_input(temp_label[0],posterior=True)
                            labels_posterior=self.add_torch_input(temp_label[1],posterior=True)
                            inputs_prior=self.add_torch_input(temp_label[2])
                            labels_prior=self.add_torch_input(temp_label[3])
                            if epoch==0:
                                #outputs1=self.PrioriModel.model(inputs_prior['contexts_tensor'])    
                                #loss_pri=self.PrioriModel.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                                pass
                            else:
                                outputs1=self.PrioriModel.model(inputs_prior['contexts_tensor'])
                                loss_pri=self.PrioriModel.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                                outputs2=self.PosteriorModel.model(inputs_posterior['contexts_tensor'])
                                loss_pos=self.PosteriorModel.calculate_loss_and_accuracy(outputs2,labels_posterior['contexts_tensor'])

                            # loss_pri.backward()
                            #if epoch<0:
                            if epoch!=0:
                                # train_ebm
                                # sampling using posterior model
                                if cfg.jsa_ebm_joint:
                                    loss_ebm_un = self.train_ebm_unlabelled(hist[i], batch, mini_pv_batch, temp_label[4])
                                loss_pri.backward()
                                loss_pos.backward()
                                if cfg.jsa_ebm_joint:
                                    #loss_ebm.backward()
                                    ebm_loss_unlabel += loss_ebm_un
                                loss=loss_pri.item()+loss_pos.item()#.to(self.device1), .to(self.device2)
                                tr_loss += loss
                                uns_loss += loss
                                #else :
                                #    loss=loss_pri.item()
                                #    tr_loss += loss_pri.item()
                                #    uns_loss += loss_pri.item()
                                uns_step+=1
                            spoken[i], _=self.get_spoken(spoken[i], batch['resp_decode'], role='system') # need to be modified
                    #supervised training
                    # comment: the mini batch settings can be modified
                    # the iteration is different from supervised learning process, 32 items in one iteration belong to the same turn
                    else:
                        if epoch!=0:
                            for i, batch in enumerate(mini_batches):
                                if dial_batch_dict['supervised']:
                                    mini_pv_batch=None if turn_num==0 else mini_pv_batches[i]
                                    inputs_prior, labels_prior = convert_batch_turn(config=cfg, cls=self.cls, 
                                    sep=self.sep, turn_batch=batch, pv_batch=mini_pv_batch, 
                                    first_turn=first_turn, posterior=False)
                                    inputs_posterior, labels_posterior = convert_batch_turn(config=cfg, cls=self.cls, 
                                    sep=self.sep, turn_batch=batch, pv_batch=mini_pv_batch, 
                                    first_turn=first_turn, posterior=True)
                                else:
                                    jsa_labels = copy.deepcopy(turn_batch['jsa_labels'][i])
                                    inputs_posterior = jsa_labels[0]
                                    labels_posterior = jsa_labels[1]
                                    inputs_prior = jsa_labels[2]
                                    labels_prior = jsa_labels[3]
                                inputs_prior = self.add_torch_input(inputs_prior)#B,T
                                labels_prior=self.add_torch_input(labels_prior)#B,T
                                inputs_posterior=self.add_torch_input(inputs_posterior,posterior=True)
                                labels_posterior=self.add_torch_input(labels_posterior,posterior=True)

                                outputs1 = self.PrioriModel.model(inputs_prior['contexts_tensor'])
                                loss_pri=self.PrioriModel.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                                loss_pri.backward()
                                if dial_batch_dict['supervised']:
                                    outputs2=self.PosteriorModel.model(inputs_posterior['contexts_tensor'])
                                    loss_pos=self.PosteriorModel.calculate_loss_and_accuracy(outputs2,labels_posterior['contexts_tensor'])
                                    loss_pos.backward()

                                # loss=loss_pri.to(self.device2)+loss_pos #.to(self.device1)
                                # loss.backward()
                                # train the ebm
                                # get_context
                                user_decode = self.PrioriModel.decode_batch(batch['user'])
                                resp_decode = self.PrioriModel.decode_batch(batch['resp'])
                                if len(spoken) < (i+1):
                                    spoken.append([])
                                    hist.append([])
                                spoken[i], hist[i]=self.get_spoken(spoken[i],user_decode, role='user')
                                if cfg.jsa_ebm_joint:
                                    loss_ebm_label = self.train_ebm_labelled(spoken[i], hist[i], batch['KB'], resp_decode)
                                    ebm_loss_label += loss_ebm_label
                                # self.bert_model

                                #update spoken
                                spoken[i], _=self.get_spoken(spoken[i], resp_decode, role='system')

                                loss = loss_pri.item() + (loss_pos.item() if dial_batch_dict['supervised'] else 0)
                                tr_loss += loss
                                sup_loss +=loss
                                sup_step +=1
                    if epoch!=0:
                        torch.nn.utils.clip_grad_norm_(self.PrioriModel.model.parameters(), 5.0)
                        torch.nn.utils.clip_grad_norm_(self.PosteriorModel.model.parameters(), 5.0)
                        torch.nn.utils.clip_grad_norm_(self.ebm.parameters(), 5.0)
                        epoch_step+=1
                        optimizer1.step()
                        optimizer1.zero_grad()
                        optimizer2.step()
                        optimizer2.zero_grad()
                        global_step+=1
                        #if cfg.use_scheduler:
                        scheduler1.step()
                        scheduler2.step()
                        if cfg.jsa_ebm_joint:
                            optimizer3.step()
                            optimizer3.zero_grad()
                            scheduler3.step()
                        if self.tb_writer:
                            self.tb_writer.add_scalar('lr1', optimizer1.param_groups[0]["lr"],global_step)
                            self.tb_writer.add_scalar('lr2', optimizer2.param_groups[0]["lr"],global_step)
                            self.tb_writer.add_scalar('lr3', optimizer3.param_groups[0]["lr"],global_step)
                            self.tb_writer.add_scalar('loss', loss, global_step)
                    pv_batch=next_pv_batch
                    #loss = 0
                    #loss_pri = 0
                    #loss_pos = 0  #clear loss to avoid out of memory
                        #torch.cuda.empty_cache()
                    
                    """
                    #sup_train, train a whole batch of 32 turns
                    if ratio_num == unlab_ratio-1:
                        train_loader = DataLoader(train_data[(batch_idx*cfg.batch_size):(batch_idx+1)*cfg.batch_size], batch_size=cfg.origin_batch_size, shuffle=True, collate_fn=train_collate_fn)
                        
                        train_post_loader = DataLoader(train_data_post[(batch_idx*cfg.batch_size):(batch_idx+1)*cfg.batch_size], batch_size=cfg.origin_batch_size, shuffle=True, collate_fn=train_collate_fn)

                        sup_loss = sup_loss + self.PrioriModel.train_one_step(train_loader,optimizer1, scheduler1)
                        sup_loss = sup_loss + self.PosteriorModel.train_one_step(train_post_loader,optimizer2, scheduler2)
                        sup_step = sup_step + 1
                    """
            logging.info('Epoch: {}, Train epoch time: {:.2f} min, loss:{:.3f}, avg_sup_loss:{:.3f}, avg_uns_loss:{:.3f}, avg_ebm_loss_label:{:.3f}, avg_ebm_loss_unlabel:{:.3f}'.format(epoch, 
                (time.time()-btm)/60, tr_loss/(epoch_step+1e-10), sup_loss/(sup_step+1e-10), uns_loss/(uns_step+1e-10), ebm_loss_label/(epoch_step+1e-10),
                ebm_loss_unlabel/(epoch_step+1e-10)))
            #eval_loss=self.PrioriModel.eval(dev_dataloader)
            if cfg.save_type =='max_score' and (epoch!=0):
                cfg.gt_db = False
                cfg.retrieve_kb = True
                eval_result=self.PrioriModel.validate_fast('test') # 'test'
                cfg.retrieve_kb = False
                cfg.gt_db = True
                bleu = eval_result['BLEU']
                f1_query = eval_result['query_f1'][2]
                qa = eval_result['qa'][2]
                logging.info('user:{:.3f}, system:{:.3f} , bleu:{:.3f}, success:{:.3f}'.format(ui, si, bleu, success))
                eval_loss = bleu/25 + qa*2 + f1_query
                logging.info('Epoch:{}, Train epoch time:{:.2f} min, epoch loss:{:.3f}, eval loss:{:.3f}'.format(epoch, (time.time()-btm)/60, tr_loss, eval_loss))
                self.tb_writer.add_scalar('eval_loss', eval_loss, epoch)
                if max_score < eval_loss:
                    max_score=eval_loss
                    self.PrioriModel.save_model()
                    self.PosteriorModel.save_model()
                    self.tokenizer.save_pretrained(cfg.exp_path)
                    # save_ebm
                    if cfg.jsa_ebm_joint:
                        ebm_save_path = os.path.join(cfg.exp_path, 'best_ebm.pt')
                        torch.save(self.ebm.state_dict(), ebm_save_path)
                else:
                    self.PrioriModel.save_model('last_model')
                    #self.PosteriorModel.save_model('last_model')
            elif cfg.save_type =='min_loss':
                if eval_loss<min_loss:
                    min_loss=eval_loss
                    self.PrioriModel.save_model()
                    self.PosteriorModel.save_model()
                    ebm_save_path = os.path.join(cfg.exp_path, 'best_ebm.pt')
                    self.tokenizer.save_pretrained(cfg.exp_path)
                    torch.save(self.ebm.state_dict(), ebm_save_path)
    # discard
    """
    def get_spoken(self,spoken,new_input, role):
        result =[]
        role_shift = {'user':' 用户：', 'system':' 客服：'}
        for i in range(len(new_input)):
            result.append(((spoken[i] if spoken!=[] else '') + role_shift[role] + new_input[i]).replace('[EOS_L]',''))
        return result
    """

    def get_spoken(self, spoken, new_input, role):
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

    def get_retrieval_prob(self, context, KB):
        return_probs = []
        tokenizer = self.bert_tokenizer
        for dial_num in range(len(context)):
            prob = 1.0
            spoken = context[dial_num]
            spoken_encoded = tokenizer.encode(spoken.replace('[EOS_SI]','')) # .replace('[UNK]','')'
            dial_KB = KB[dial_num]
            KB_result =[] 
            KB_query = []
            candidates = []
            # avoid special tokens in KB
            flag = 0
            for token in special_tokens:
                if token in dial_KB:
                    flag=1
            if flag==0:
                for triple in dial_KB.split('；'):
                    if triple!='':
                        KB_query.append({'context':spoken_encoded,'triple':tokenizer.encode(triple)})
            else:
                prob = math.exp(-100)
            if KB_query!=[]:
                batch = collate_fn(KB_query)
                if cfg.only_one_model:
                    #candidate = copy.deepcopy(batch['input'])
                    for key, val in batch.items():
                        batch[key] = batch[key].to(self.device2)
                    with torch.no_grad():
                        logits = self.bert_model(input_ids=batch["input"], attention_mask=batch["input_attention"], token_type_ids=batch["input_type"]).logits
                        probs = F.softmax(logits, dim=1)
                        predicts=(probs[:,0]).cpu().tolist() # because 0 in the NSP task means consistency
                    for num in range(len(predicts)):
                        prob = prob * predicts[num]
            return_probs.append(copy.deepcopy(math.log(prob)))
        return return_probs

    def train_ebm_unlabelled(self, contexts, batch, pv_batch, kb_sequences):
        # beam size is the proposal number for training ebm
        BEAM_SIZE = 4
        resps = batch['resp_decode']
        #kb_tokens = batch['db_decode']
        return_probs = []
        tokenizer = self.bert_tokenizer
        loss_ebm = 0.0
        cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
        eos_db_id=self.tokenizer.convert_tokens_to_ids('[EOS_K]')
        sep_id=self.tokenizer.convert_tokens_to_ids('[SEP]')
        max_len_k=80
        # convert_eval_batch_turn deals with a batch input
        gen_contexts = convert_eval_batch_turn(config=cfg, cls=cls, turn_batch=turn_batch, pv_batch=pv_batch,
         mode='gen_kb', posterior=True)
        #gen_contexts.to(self.device2)
        db_batches = self.PosteriorModel.generate_batch(self.PosteriorModel.model, gen_contexts, max_len_k, eos_db_id, beam=BEAM_SIZE, return_prob=True)
        gt_input = get_retrieval_sequence(tokenizer, contexts, kb_sequences)
        gt_input.to(self.device3)
        gt_logits = self.ebm(input_ids=gt_input['input_ids'], attention_mask=gt_input["attention_mask"])
        if (torch.isnan(gt_logits).any() or torch.isinf(gt_logits).any()) and self.nan_num<5:
            print(f"nan found in unsup training: {gt_logits}")
            print(contexts)
            print(kb_sequences)
            self.nan_num = self.nan_num + 1
            loss = 0.0
        else:
            loss = -sum(gt_logits)

        for dial_num in range(len(contexts)):
            # prob = 1.0
            # dialog history
            context = contexts[dial_num]
            db_batch = db_batches[dial_num]
            spoken_encoded = tokenizer.encode(context.replace('[EOS_SI]','')) # .replace('[UNK]','')'
            # avoid special tokens in KB
            # getting the samples ready for training ebm
            flag = 0
            #if kb_sequence!='': 

            # generating proposals and get their probabilities

            # gen_hidden
            #db_batch, db_probs = self.PosteriorModel.generate_batch(self.PosteriorModel, [gen_context], max_len_k, eos_db_id, beam=BEAM_SIZE, return_prob=True)
            if len(db_batch)>0:
                db_pregen =  [] # tmp[0] for tmp in db_batch
                db_logits =  [] # tmp[1] for tmp in db_batch
                for tmp_num in range(len(db_batch)):
                    if db_batch[tmp_num][1]>-20.0: # remove those with low sampling probs
                        db_pregen.append(db_batch[tmp_num][0])
                        db_logits.append(db_batch[tmp_num][1])
            if len(db_pregen)>0:
                db_gen = self.PosteriorModel.get_xspn(db_pregen, eos_db_id, sep_id)
                db_decode = self.PosteriorModel.decode_batch(db_gen)
                propose_input = get_retrieval_sequence(tokenizer, [context]*len(db_decode), db_decode)
                propose_input.to(self.device3)
                logits = self.ebm(input_ids=propose_input['input_ids'], attention_mask=propose_input["attention_mask"])
                # combine logits with probability to get the final loss
                is_ratio = []
                for j in range(len(db_logits)):
                    if (logits[j].item())<-100.0:
                        is_ratio.append(math.exp(-100.0))
                        print(f"large is ratio found in unsup training:{db_logits[j]},{logits[j].item()}")
                        #nan_num = nan_num + 1
                    elif (logits[j].item())>100.0:
                        is_ratio.append(math.exp(100.0))
                        print(f"large is ratio found in unsup training:{db_logits[j]},{logits[j].item()}")
                    else:
                        if cfg.residual:
                            is_ratio.append(math.exp (logits[j].item()))
                        else:
                            is_ratio.append(math.exp(logits[j].item()))
                            # math.exp(-db_logits[j] + logits[j].item()
                normalize = sum(is_ratio)
                if normalize>0.0:
                    for j in range(len(db_logits)):
                        loss = loss + is_ratio[j]*logits[j]/normalize
        if loss!=0.0:
            if (torch.isnan(loss).any() or torch.isinf(loss).any()) and self.nan_num<5:
                print(f"nan found in neg sample in unsup training: {logits}")
                print(contexts)
                print(kb_sequences)
                self.nan_num = self.nan_num + 1
                loss = 0.0
            else:
                loss.backward()
                loss_ebm += loss.item()
        
        return loss_ebm

    def train_ebm_labelled(self, spokens, contexts, KB, resps):
        tokenizer = self.bert_tokenizer
        loss = 0.0
        for dial_num in range(len(contexts)):
            # prob = 1.0
            spoken = spokens[dial_num]
            context = contexts[dial_num]
            resp = resps[dial_num]
            spoken_encoded = tokenizer.encode(spoken.replace('[EOS_SI]','')) # .replace('[UNK]','')'
            dial_KB = KB[dial_num]
            KB_result =[]
            # avoid special tokens in KB
            # getting the samples ready for training ebm
            # refer to get_retrieval_data(data, tokenizer, dial_ids=None, ebm=True) for the origin implementation
            # todo: make this section a function: get_turn_level_ebm
            negative = []
            positive = []
            flag = 0
            for _,kvs in dial_KB.items():
                for k,v in  kvs.items():
                    if k !='type': # remove type for now
                        if (',' not in v) and (v!= ''):
                            if (v not in spoken) and (v not in resp):
                                negative.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ v))
                            elif v in resp:
                                positive.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ v))
                                #if v in resp:
                                #    positive.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ v))
                                #else:
                                #    negative.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ v))
                            #elif v in resp:
                            #    positive.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ v))
                        else:
                            tmp_v = []
                            flag = 0
                            for value in v.split(','):
                                if (value not in spoken) and (value not in resp):
                                    tmp_v.append(value)
                                elif value in resp:
                                    tmp_v.append(value)
                                    flag = 1
                                #elif value in resp:
                                #    tmp_v.append(value)
                                #    flag = 1
                            if (flag == 0) and (tmp_v!=[]):
                                negative.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ ','.join(tmp_v)))   
                            elif flag == 1:
                                positive.append(tokenizer.encode(k.replace('name','名称').replace('type','类型') +':'+ ','.join(tmp_v)))
            #hist, hist_seq = get_dial_hist(hist, turn)
            #seq = tokenizer.encode(hist_seq)
            context_encoded = tokenizer.encode(context.replace('[EOS_SI]',''))
            turn_cases = []
            for p in positive:
                turn_cases.append({'context':context_encoded, 'triple':p, 'label':1})
            for n in negative:
                turn_cases.append({'context':context_encoded, 'triple':n, 'label':-1})
            if turn_cases!=[]:
                #turn_case = {'context':spoken, 'cases':turn_cases}
                #cases.append(turn_case)
                #try:  # avoid OOM
                #positive_count = []
                #for cases in batch['cases']:
                gt_seq = ''
                tmp_count = 0
                #for c in cases:
                for c in turn_cases:
                    if c['label']==1:
                        gt_seq = gt_seq + tokenizer.decode(c['triple']).replace('[CLS]','').replace('[SEP]','') + '；'
                        tmp_count += 1 
                if '；' in gt_seq:
                    gt_seq = gt_seq[:-1] 
                # training 
                gt_input = get_retrieval_sequence(tokenizer, [context], [gt_seq])
                gt_input.to(self.device3)
                logits = self.ebm(input_ids=gt_input['input_ids'], attention_mask=gt_input["attention_mask"])
                if (torch.isnan(logits).any() or torch.isinf(logits).any()) and self.nan_num<5:
                    print(f"nan found in sup training:{logits.item()}")
                    self.nan_num = self.nan_num + 1
                    break
                else:
                    loss = loss - sum(logits)
                p_batch = collate_fn(turn_cases)
                for key, val in p_batch.items():
                    if type(p_batch[key]) is list:
                        continue
                    p_batch[key] = p_batch[key].to(self.device2)
                if p_batch!={}:
                    with torch.no_grad():
                        org_logits = self.bert_model(input_ids=p_batch["input"], attention_mask=p_batch["input_attention"], token_type_ids=p_batch["input_type"]).logits
                        probs = F.softmax(org_logits, dim=1)
                        accept_prob = probs[:,0].cpu().tolist() # 0 means coherency in bert pretraining
                        gt_label = p_batch['label'].cpu().tolist() 
                else:
                    accept_prob = []
                triple_num = len(accept_prob)
                # sampling, mis mode not supported yet
                if cfg.train_ebm_mis and 'cached' in turn_cases[0]:
                    proposals = [turn_cases[0]['cached'][0]]
                    proposal_log_probs = [turn_cases[0]['cached'][1]]
                    proposal_wrong_num = [turn_cases[0]['cached'][2]]
                else:
                    proposals = []
                    proposal_log_probs = []
                    proposal_wrong_num = []
                # debugging only 
                #print(cfg.train_sample_num)
                for sample_num in range(cfg.train_sample_num):
                    p_prob = 0.0
                    proposal = []
                    proposal_id = []
                    for num in range(triple_num):
                        p = random.random()
                        if p< accept_prob[num]:
                            proposal.append(tokenizer.decode(turn_cases[num]['triple'][1:-1]).replace(' ','')) # can be unified to the .replace('[CLS]','').replace('[SEP]','')
                            # can directly concatenate all the triples to improve efficiency
                            p_prob += math.log(accept_prob[num])
                            proposal_id.append(0)
                        else:
                            p_prob += math.log(1-accept_prob[num])
                            proposal_id.append(1)
                    if proposal_id!=gt_label or cfg.use_all_proposal: #use cfg.use_all_proposal to include gt_label to be trained
                        proposals.append('；'.join(proposal))
                        proposal_log_probs.append(p_prob)
                        proposal_wrong_num.append(sum(gt_label[i]!=proposal_id[i] for i in range(len(gt_label))))
                # get IS_loss, avoiding OOM
                is_logits = []
                sample_num = len(proposals)
                positive_count = torch.tensor([float(len(item.split('；'))) for item in proposals], 
                dtype=torch.float).to(self.device3)
                if sample_num>0:
                    input = get_retrieval_sequence(tokenizer, [context]*sample_num, proposals)
                    input.to(self.device3)
                    if cfg.add_extra_feature:
                        is_logits.extend(self.ebm(input_ids=input['input_ids'], attention_mask=input["attention_mask"], feature=positive_count))
                    else:
                        is_logits.extend(self.ebm(input_ids=input['input_ids'], attention_mask=input["attention_mask"]))
                    is_ratio = []
                    for j in range(sample_num):
                        if (-proposal_log_probs[j] + is_logits[j].item())>100:
                            is_ratio.append(math.exp(100))
                            print(f"large is ratio found in sup training:{proposal_log_probs[j]},{is_logits[j].item()}")
                        if (-proposal_log_probs[j] + is_logits[j].item())<-100:
                            is_ratio.append(math.exp(-100))
                            print(f"large is ratio found:{proposal_log_probs[j]},{is_logits[j].item()}")
                        else:
                            if cfg.residual:
                                is_ratio.append(math.exp (is_logits[j].item()))
                            else:
                                is_ratio.append(math.exp(-proposal_log_probs[j] + is_logits[j].item()))
                    if cfg.train_ebm_mis:
                        mis_results = {}
                        max = is_ratio[0]
                        current = 0
                        lengths = 0
                        #mis_results[0] = 0
                        for j in range(sample_num):
                            tmp_prob = random.random()
                            if tmp_prob<(is_ratio[j]/max): # is_ratio[j]> max,
                                # actually max is not necessarily the current max
                                mis_results[current] = lengths
                                max = is_ratio[j]
                                current = j
                                lengths = 1
                            else:
                                lengths += 1
                        #if current==0:
                        #    mis_results[0] = lengths
                        mis_results[current] = lengths
                        # sample should be added
                        normalize = sum(mis_results[tmp] for tmp in mis_results) 
                        # mis performs averaging instead of weighted averaging
                        turn_cases[0]['cached']=(proposals[j], proposal_log_probs[j], proposal_wrong_num[j])
                        # save cached results
                    else:
                        normalize = sum(is_ratio)
                    if normalize>0.0:
                        if cfg.train_ebm_mis:    
                            for index, length in mis_results.items():
                                loss = loss + (length*is_logits[index])/normalize
                        else:
                            for j in range(sample_num):
                                if cfg.reject_control:
                                    if is_ratio[j]/normalize>0.003: # add reject control
                                        loss = loss + (is_ratio[j]* is_logits[j])/normalize
                                    else:
                                        if random.random()<is_ratio[j]*200/normalize:
                                            loss = loss + 0.005*is_logits[j]
                                else:
                                    loss = loss + is_ratio[j]*is_logits[j]/normalize
        if loss != 0.0:
            loss.backward()
            return loss.item()
        else:
            return 0.0

    def get_retrieval_prob_ebm(self, context, KB):
        return_probs = []
        tokenizer = self.bert_tokenizer
        for dial_num in range(len(context)):
            # prob = 1.0
            spoken = context[dial_num]
            spoken_encoded = tokenizer.encode(spoken.replace('[EOS_SI]','')) # .replace('[UNK]','')'
            dial_KB = KB[dial_num]
            KB_result =[] 
            KB_query = []
            candidates = []
            # avoid special tokens in KB
            flag = 0
            for token in special_tokens:
                if token in dial_KB:
                    flag=1
            if flag==0:
                input = tokenizer((spoken + '[SEP]' + dial_KB).replace('[UNK]','g'),
                 return_tensors="pt", padding=True, truncation=True, max_length=512)
                input.to(self.device3)
                with torch.no_grad():
                    ebm_logits = self.ebm(input_ids=input['input_ids'], attention_mask=input["attention_mask"])
                    return_probs.append(ebm_logits.cpu().tolist()[0][0]) 
                #tokenizer.encode((spoken + '[SEP]' + dial_KB).replace('[UNK]','g'))
                #for triple in dial_KB.split('；'):
                #    if triple!='':
                #        KB_query.append({'context':spoken_encoded,'triple':tokenizer.encode(triple)})
            else:
                #prob = math.exp(-100)
                return_probs.append(-300.0) # the num can be refined
            #if KB_query!=[]:
            #prob = math.exp(logits)
            """
            batch = collate_fn(KB_query)
            if cfg.only_one_model:
                #candidate = copy.deepcopy(batch['input'])
                for key, val in batch.items():
                    batch[key] = batch[key].to(self.device2)
                logits = self.bert_model(input_ids=batch["input"], attention_mask=batch["input_attention"], token_type_ids=batch["input_type"]).logits
                probs = F.softmax(logits, dim=1)
                predicts=(probs[:,0]).cpu().tolist() # because 0 in the NSP task means consistency
                for num in range(len(predicts)):
                    prob = prob * predicts[num]
            """
            #return_probs.append(copy.deepcopy(math.log(prob)))
        return return_probs

    def get_jsa_prob(self,logits_pri,logits_post,labels_pri,labels_post):
        # logits_pri:B,T1,V
        # logits_post:B,T2,V
        # labels_pri:B,T1. uspn's,bspn's label in prior sequence
        # labels_post:B,T2. bspn's label in posterior sequence
        # what labels do is to find the logits corresponding to bspn
        prob=[]
        for dial_idx in range(logits_pri.size(0)):
            label_pri=labels_pri[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist() #pad_id处为0，bspn为1
            label_post=labels_post[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist()
            h_len_post=len(label_post)-label_post[::-1].index(1)-label_post.index(1)
            h_len_pri=len(label_pri)-label_pri[::-1].index(1)-label_pri.index(1)
            idx1=label_pri.index(1)
            idx2=label_post.index(1)
            probs_pri=F.softmax(logits_pri[dial_idx, idx1:idx1+h_len_pri-1,:],dim=-1)
            probs_post=F.softmax(logits_post[dial_idx, idx2:idx2+h_len_post-1,:],dim=-1)
            up=torch.tensor(0.0)
            down=torch.tensor(0.0)
        
            for up_num in range(probs_pri.size()[0]):#loc2-loc1-1
                #if probs_pri.size()[0]!=loc2-loc1-1
                #    print(probs_pri.size()[0])
                #    print(loc2-loc1-1)
                if probs_pri[up_num,labels_pri[dial_idx,idx1+up_num+1]]!=0:
                    up=up+math.log(probs_pri[up_num,labels_pri[dial_idx,idx1+up_num+1]])#probs_pri[up_num,:].max()
                else:
                    up=up-1000
            for down_num in range(probs_post.size()[0]):#loc4-loc3-1
                if probs_post[down_num,labels_post[dial_idx,idx2+down_num+1]]!=0:
                    down=down+math.log(probs_post[down_num,labels_post[dial_idx,idx2+down_num+1]])#probs_pri[down_num,labels_pri[logits_pri.size(1)-loc2+up_num]]
                else:
                    down=down-1000
            prob.append(up.item()-down.item())
        return prob    

    def add_torch_input(self, inputs, posterior=False):
        # to tensor and to device
        if 'contexts_np' not in inputs:
            inputs['contexts_np'],_=padSeqs_gpt(inputs['contexts'], cfg.pad_id)
        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        if posterior:
            contexts_tensor = contexts_tensor.to(self.device2)
        else:
            contexts_tensor = contexts_tensor.to(self.device1)
        inputs['contexts_tensor'] = contexts_tensor
        return inputs   

def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')
    if not os.path.exists('./log'):
        os.mkdir('./log')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()
    cfg.mode = args.mode
    parse_arg_cfg(args)
    if cfg.exp_path=='':
        experiments_path = './experiments'
        cfg.exp_path = os.path.join(experiments_path, cfg.exp_name)
        if not os.path.exists(cfg.exp_path):
            os.mkdir(cfg.exp_path)

    cfg._init_logging_handler()

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    # initialize model
    m = Model(cfg.device)
    # train
    if cfg.mode=='train':
        m.train()
    if cfg.mode=='pretrain':
        m.train()
    if cfg.mode=='train_post':
        m = Model(cfg.device,posterior=True)
        m.train()
    if cfg.mode=='train_jsa':
        semi = Semi_supervision(cfg)
        semi.jsa_train()
    if cfg.mode=='train_jsa_ebm':
        semi = Semi_supervision(cfg)
        semi.jsa_train_ebm()
    if cfg.mode=='test':
        m.validate_fast(data = 'test')
        #m.test_end_to_end()
    if cfg.mode=='test_post':
        m = Model(cfg.device,posterior=True)
        m.validate_post(data = 'test')
    if cfg.mode=='generate_post': # generate pseudo label
        m = Model(cfg.device,posterior=True)
        m.generate_pseudo_label()

 
if __name__ == "__main__":
    main()

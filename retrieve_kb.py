"""
Copyright 2024 Tsinghua University
Author: Yucheng Cai (cyc22@mails.tsinghua.edu.cn)
"""

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertForPreTraining, BertForNextSentencePrediction
from transformers import BertTokenizer
from reader import *
from metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, CrossEntropyLoss, CosineEmbeddingLoss, CosineSimilarity
from torch.utils.data import DataLoader
from sentence_transformers import models, SentenceTransformer, losses, datasets, util

import os, shutil
import random
import argparse
import time
import logging
import json
from tqdm import tqdm
import numpy as np
import copy, re
from torch.utils.tensorboard import SummaryWriter
from config import global_config as cfg
from util import get_sep_optimizers

api_dict = {}

def collate_fn(batch):
    #print(batch)
    pad_result = {}
    for key in batch[0]:# same pad_id for all values
        if key =='input' or key =='attention_mask' or key =='slot':    
            max_len = max(len(input[key]) for input in batch)
            pad_batch=np.ones((len(batch), max_len))*cfg.pad_id  #-100
            for idx, s in enumerate(batch):
                pad_batch[idx, :len(s[key])] = s[key]
            pad_result[key] = torch.from_numpy(pad_batch).long()
        if key =='id' or key =='turn_num' or key =='local_kb' or key =='api_label'or key =='labels': # labels
            pad_res=[]
            for _, s in enumerate(batch):
                pad_res.append(s[key])
            pad_result[key] = pad_res

        if key =='domain': # labels
            pad_batch=np.ones(len(batch))
            for idx, s in enumerate(batch):
                pad_batch[idx] = s[key]
            pad_result[key] = torch.from_numpy(pad_batch).long()
    return pad_result

class Api_model(torch.nn.Module):
    def __init__(self, cfg, tokenizer):
        super(Api_model, self).__init__()
        self.cfg = cfg
        
        self.bert_model=BertModel.from_pretrained(cfg.api_encoder_path)#cfg.model_path
        self.bert_model.resize_token_embeddings(len(tokenizer))
        self.dropout = Dropout(cfg.dropout)
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, cfg.num_apis) 
        self.num_labels = cfg.num_apis

    def forward(self,input_ids: torch.tensor,
                attention_mask: torch.tensor,
                label=None):
        hidden_states = self.bert_model(input_ids=input_ids,attention_mask = attention_mask)[0]
        #slot_logits = self.slot_classifier(self.dropout(hidden_states))
        loss_fct = CrossEntropyLoss()
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
        #else:
        #slot_loss = loss_fct(slot_logits.view(-1, self.num_slots), slot_labels.view(-1).type(torch.long))

        #pooled_output =  hidden_states[:, :].mean(dim=1)
        # modified the pooled_output setting to avoid scale difference
        pooled_output =  (hidden_states[:, :, :]*attention_mask.unsqueeze(-1)).sum(dim=1)/attention_mask.sum(dim=1).unsqueeze(-1)
        logits = self.classifier(self.dropout(pooled_output))

        if label!=None:
            loss = loss_fct(logits.view(-1, self.num_labels), label.type(torch.long))
        
        if label!=None:
            return logits, loss
        else:
            return logits

def train_api_model(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg.api_encoder_path)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    model = Api_model(cfg, tokenizer)
    model.to(cfg.device[0])#is list
    cfg.batch_size = 10
    cfg.epoch_num = 40

    encoded_data = read_data_api(tokenizer)
    
    #train_dataset = encoded_data['train']
    #train_dataset.extend(encoded_data['dev'])
    train_dataset = encoded_data['dev']
    train_dataset.extend(encoded_data['train'])
    print(len(train_dataset))
    train_dataloader=DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn)
    test_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn)
    optimizer, scheduler = get_sep_optimizers(cfg, num_turns=len(encoded_data['train']), model=model)
    global_step = 0
    metrics_to_log = {}
    best_score = -1
    for epoch in range(cfg.epoch_num):
        model.train()
        epoch_loss = 0
        epoch_step = 0
        num_batches = 0

        for batch in tqdm(train_dataloader):
            num_batches += 1

            # Transfer to gpu
            if torch.cuda.is_available():
                for key, val in batch.items():
                    if type(batch[key]) is list:
                        continue
                    batch[key] = batch[key].to(cfg.device[0])
                loss = model(input_ids=batch["input"], attention_mask=batch["attention_mask"],label=batch["domain"])[1]
                loss.backward()
                epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                epoch_step += 1
                if epoch_step % cfg.gradient_accumulation_steps == 0 or num_batches==len(train_dataloader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

        logging.info("Epoch loss: {}".format(epoch_loss / num_batches))

        # Evaluate and save checkpoint
        domain_acc, qa, api, total, total_e = evaluate(model, test_dataloader, cfg)
        print(f"domain_acc: {domain_acc}, qa: {qa}, api: {api}, total: {total}, total_e: {total_e}")
        score = qa[2] + api[2]
        metrics_to_log["eval_score"] = score
        logging.info("score: {}".format(score))
        if score > best_score:
            logging.info("New best results found! Score: {}".format(score))
            #model.bert_model.save_pretrained(cfg.save_dir)
            if not os.path.exists(cfg.api_save_dir):
                os.mkdir(cfg.api_save_dir)
            torch.save(model.state_dict(), os.path.join(cfg.api_save_dir, "model.pt"))
            best_score = score
    model.load_state_dict(torch.load(os.path.join(cfg.api_save_dir, "model.pt")))
    domain_acc = evaluate(model, test_dataloader, cfg)
    score = domain_acc
    print(score)

class Retrieval_model(torch.nn.Module):
    def __init__(self, cfg, tokenizer):
        super(Retrieval_model, self).__init__()
        self.cfg = cfg
        
        self.bert_model=BertModel.from_pretrained(cfg.api_encoder_path)#cfg.model_path
        self.bert_model.resize_token_embeddings(len(tokenizer))

    # input_ids=batch["input"], attention_mask=batch["attention_mask"], label=batch["domain"], global_index=global_index
    def forward(self,input_ids: torch.tensor,
                attention_mask: torch.tensor,
                indices,
                topk,
                label=None,
                ):
        loss = 0.0
        bsz = input_ids.shape[0]
        hidden_states = self.bert_model(input_ids=input_ids,attention_mask = attention_mask)[0]
        #slot_logits = self.slot_classifier(self.dropout(hidden_states))
        loss_fct = CrossEntropyLoss()
        # B, V
        
        pooled_output =  (hidden_states[:, :, :]*attention_mask.unsqueeze(-1)).sum(dim=1)/attention_mask.sum(dim=1).unsqueeze(-1)
        
        logits = []
        #probs = F.softmax(logits, dim=-1)
        #logits = self.classifier(self.dropout(pooled_output))
        for i in range(bsz):
            if label!=None:
                if label[i]!=[]:
                    index = torch.tensor(indices[i]).to(cfg.device[0])
                    logit = index @ pooled_output[i, : ]
                    returned_logit = torch.topk(logit, k=topk)
                    logits.append(returned_logit)
                    for l in label[i]:
                        if l!=-1:
                            tmp_l = torch.tensor(l).to(cfg.device[0])
                            loss += loss_fct(logit, tmp_l.type(torch.long))
            else:
                index = torch.tensor(indices[i]).to(cfg.device[0])
                logit = index @ pooled_output[i, : ]
                returned_logit = torch.topk(logit, k=topk)
                logits.append(returned_logit)
        if label!=None:
            return logits, loss
        else:
            return logits

def train_api_model_retrieve(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg.api_encoder_path)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    #passage_model = BertModel.from_pretrained(cfg.api_encoder_path) # index_model   
    passage_model = SentenceTransformer(cfg.sentence_encoder_path)
    passage_model.eval()

    context_model = Retrieval_model(cfg, tokenizer)

    # passage_model.to(cfg.device[0])#is list
    # train the context_model only
    context_model.to(cfg.device[0])
    cfg.batch_size = 8
    cfg.epoch_num = 40
    
    global_index = build_global_index(passage_model, tokenizer, batch_size=16)

    encoded_data = read_data_api(tokenizer, ret_train=True)
    
    train_dataset = encoded_data['train']
    #train_dataset.extend(encoded_data['dev'])
    #train_dataset = encoded_data['dev']
    #train_dataset.extend(encoded_data['train'])
    print(len(train_dataset))
    train_dataloader=DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn)
    test_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn)
    optimizer, scheduler = get_sep_optimizers(cfg, num_turns=len(encoded_data['train']), model=context_model)
    global_step = 0
    oom_time = 0
    metrics_to_log = {}
    best_score = -1
    for epoch in range(cfg.epoch_num):
        context_model.train()
        epoch_loss = 0
        epoch_step = 0
        num_batches = 0

        for batch in tqdm(train_dataloader):
            num_batches += 1
            try:
            # Transfer to gpu
                # if torch.cuda.is_available():
                for key, val in batch.items():
                    if type(batch[key]) is list:
                        continue
                    batch[key] = batch[key].to(cfg.device[0])

                local_index = build_local_index(passage_model, batch['local_kb'], global_index) # passage_model, tokenizer, user_info, global_embedding
                loss = context_model(input_ids=batch["input"], attention_mask=batch["attention_mask"], indices=local_index, topk=20, label=batch["labels"])[1]
                
                if loss!=0.0:
                    loss.backward()
                    epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(context_model.parameters(), 5.0)
                epoch_step += 1
                if epoch_step % cfg.gradient_accumulation_steps == 0 or num_batches==len(train_dataloader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    logging.info("WARNING: ran out of memory,times: {}, batch size: {}".format(oom_time, cfg.batch_size))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logging.info(str(exception))
                    raise exception
        logging.info("Epoch loss: {}".format(epoch_loss / num_batches))

        # Evaluate and save checkpoint
        r_1, r_5, r_20  = evaluate_ret(passage_model, context_model, dev_dataloader, global_index, cfg)
        print(f"Recall@1: {r_1}, Recall@5: {r_5}, Recall@20: {r_20}")
        score = r_1 + r_5 + r_20
        metrics_to_log["eval_score"] = score
        logging.info("score: {}".format(score))
        if score > best_score:
            logging.info("New best results found! Score: {}".format(score))
            #model.bert_model.save_pretrained(cfg.save_dir)
            if not os.path.exists(cfg.apiret_save_dir):
                os.mkdir(cfg.apiret_save_dir)
            torch.save(context_model.state_dict(), os.path.join(cfg.apiret_save_dir, "model.pt"))
            best_score = score
    context_model.load_state_dict(torch.load(os.path.join(cfg.apiret_save_dir, "model.pt")))
    r_1, r_5, r_20 = evaluate_ret(passage_model, context_model, dev_dataloader, global_index, cfg)
    print(f"Recall@1: {r_1}, Recall@5: {r_5}, Recall@20: {r_20}")
    score = r_1 + r_5 + r_20
    print(score)

def test_api_model_retrieve(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg.api_encoder_path)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    passage_model = SentenceTransformer(cfg.sentence_encoder_path)
    passage_model.eval()

    encoded_data = read_data_api(tokenizer, ret_train=True)
    test_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn)

    global_index = build_global_index(passage_model, tokenizer, batch_size=16)

    context_model = Retrieval_model(cfg, tokenizer)
    context_model.load_state_dict(torch.load(os.path.join(cfg.apiret_save_dir, "model.pt")))
    context_model.to(cfg.device[0])

    r_1, r_5, r_20  = evaluate_ret(passage_model, context_model, test_dataloader, global_index, cfg)
    print(f"Recall@1: {r_1}, Recall@5: {r_5}, Recall@20: {r_20}")
    score = r_1 + r_5 + r_20
    #metrics_to_log["eval_score"] = score
    logging.info("score: {}".format(score))
    return

def build_global_index(passage_model, tokenizer, batch_size=16):
    passage_model.eval()
    passages = json.loads(open('kb/ret_qa.json', 'r').read())
    n_batch = int(len(passages)/batch_size) + 1

    total = 0
    index = []
    with torch.no_grad():
        for i in range(n_batch):
            batch = passages[i * batch_size : (i + 1) * batch_size]
            """
            batch_enc = tokenizer(
                batch,
                padding="longest",
                return_tensors="pt",
                max_length=cfg.max_length,
                truncation=True,
            )
            """
            embeddings = passage_model.encode(batch, show_progress_bar=False)
            index.extend(embeddings)
            #embeddings = passage_model(**_to_cuda(batch_enc), is_passages=True)
            #index.embeddings[:, total : total + len(embeddings)] = embeddings.T
            total += len(embeddings)
            if i % 500 == 0 and i > 0:
                print(f"Number of passages encoded: {total}")
                #logger.info(f"Number of passages encoded: {total}")

    return index

def build_local_index(passage_model, user_infos, global_embedding):
    passage_model.eval()
    #n_batch = int(len(passages)/batch_size) + 1

    total = 0
    indices = []
    with torch.no_grad():
        #for i in range(n_batch):
        #batch = passages[i * batch_size : (i + 1) * batch_size]
        for i in range(len(user_infos)):
            info = user_infos[i]
            if info!=[]:
                local_embeddings = passage_model.encode(info, show_progress_bar=False)
                tmp = copy.deepcopy(global_embedding)
                tmp.extend(local_embeddings)
                indices.append(np.array(tmp))
            else:
                indices.append(np.array(global_embedding))
        # local_embeddings should be appended to the global_embedding

    return indices

def evaluate(model, dev_dataloader, cfg):
    model.eval()
    preds = []
    trues = []
    data_dict = json.loads(open('data/test.json', 'r').read())
    #data_dict = json.loads(open('data/test.json', 'r').read())
    wrong_domain = []
    """
    data_dict = {}
    for dial in test_data:
        id = dial['id'] 
        data_dict[id] = dial
    """
    for batch in dev_dataloader:#tqdm(dev_dataloader, desc="Evaluating"):
        with torch.no_grad():
            for key, val in batch.items():
                if type(batch[key]) is list:
                    continue
                batch[key] = batch[key].to(cfg.device[0])
                
            logits, _ = model(input_ids=batch["input"],attention_mask=batch["attention_mask"],label=batch["domain"])
            """
            slot_preds = torch.argmax(slot_logits, dim=2).detach().cpu().numpy()
            actual_predicted_slots = slot_preds.squeeze().tolist()
            actual_gold_slots = batch["slot"].cpu().numpy().squeeze().tolist()

            pad_ind = len(batch["attention_mask"].tolist()[0])#batch["attention_mask"].tolist()[0].index(0)
            # actual_gold_slots = actual_gold_slots[1:pad_ind - 1]
            # actual_predicted_slots = actual_predicted_slots[1:pad_ind - 1]
            pred_slots += actual_predicted_slots
            true_slots += actual_gold_slots
            # probs of 4 need to be reduced, 5.0 is a 
            logits[:,4] = logits[:,4]-6.0
            logits[:,3] = logits[:,3]
            """
            intent_preds = torch.argmax(logits, dim=1).cpu().tolist()
            probs = F.softmax(logits,dim=-1)
            prob = []
            for i in range(len(intent_preds)):
                prob.append(probs[i,intent_preds[i]].cpu())
            pred = intent_preds
            true = batch["domain"].cpu().tolist()
            preds += pred
            trues += true
            for i in range(len(pred)):
                if pred[i]!=true[i]:
                    turn_id = batch['id'][i]
                    turn_num = batch['turn_num'][i]
                    if len(data_dict[turn_id])>turn_num :#or (turn_num in data_dict[turn_id])
                        turn = data_dict[turn_id][turn_num]
                        wrong_domain.append((turn, pred[i]))
                """
                if actual_predicted_slots[i]!=actual_gold_slots[i]:
                    turn_id = batch['id'][i]
                    turn_num = batch['turn_num'][i]
                    turn = data_dict[turn_id]['log'][turn_num]
                    wrong_slot.append((turn, [1 in actual_predicted_slots[i], 3 in actual_predicted_slots[i],
                     5 in actual_predicted_slots[i], 7 in actual_predicted_slots[i]]))
                """
    #data.extend(json.loads(open('data/eval_new.json', 'r').read()))
    #data.extend(json.loads(open('data/test_new.json', 'r').read()))
    
    json.dump(wrong_domain, open('wrong_domain.json', 'w'), indent=2, ensure_ascii=False)
    #json.dump(wrong_slot, open('wrong_slot.json', 'w'), indent=2, ensure_ascii=False)
    domain_acc = sum(p == t for p, t in zip(preds, trues)) / len(preds)
    tp_q = sum((p == 3 and  t==3) for p, t in zip(preds, trues))
    fp_q = sum((p == 3 and  t!=3) for p, t in zip(preds, trues))
    fn_q = sum((p != 3 and  t==3) for p, t in zip(preds, trues))

    tp_api = sum((p == t and p in [0, 1, 2]) for p, t in zip(preds, trues))
    fp_api = sum((p != t and  p in [0, 1, 2] and t in [3, 4]) for p, t in zip(preds, trues))
    fn_api = sum((p != t and  t in [0, 1, 2]) for p, t in zip(preds, trues))

    tp_total = sum((p == t and p in [0, 1, 2, 3]) for p, t in zip(preds, trues))
    fp_total = sum((p != t and  t in [0, 1, 2, 3] and p==4) for p, t in zip(preds, trues))
    fn_total = sum((p != t and  p in [0, 1, 2, 3]) for p, t in zip(preds, trues))

    tp_total_e = sum(((t in [0, 1, 2, 3]) and (p in [0, 1, 2, 3])) for p, t in zip(preds, trues))
    fp_total_e = sum((t==4 and  p in [0, 1, 2, 3]) for p, t in zip(preds, trues))
    fn_total_e = sum((p==4 and  t in [0, 1, 2, 3]) for p, t in zip(preds, trues))

    e = 1e-5
    P_q, P_api, P_total, P_total_e = tp_q/(tp_q + fp_q + e), tp_api/(tp_api + fp_api + e), tp_total/(tp_total + fp_total + e), tp_total_e/(tp_total_e + fp_total_e + e)
    R_q, R_api, R_total, R_total_e = tp_q/(tp_q + fn_q + e), tp_api/(tp_api + fn_api + e), tp_total/(tp_total + fn_total + e), tp_total_e/(tp_total_e + fn_total_e + e)
    F1_q, F1_api = 2*P_q*R_q/(P_q + R_q +e), 2*P_api*R_api/(P_api + R_api + e)
    F1_total, F1_total_e = 2*P_total*R_total/(P_total + R_total +e), 2*P_total_e*R_total_e/(P_total_e + R_total_e +e)

    domain_acc = sum(p == t for p, t in zip(preds, trues)) / len(preds)
    """
    slot_acc = 0
    slot_turn = 0
    for num in range(len(true_slots)):
        s = true_slots[num]
        p = pred_slots[num]
        index = [1 in s, 3 in s, 5 in s, 7 in s]
        index_p = [1 in p, 3 in p, 5 in p, 7 in p]
        if sum(index)!=0 or sum(index_p)!=0:
            slot_turn = slot_turn + 1
            if index==index_p:
                slot_acc = slot_acc + 1
    slot_acc = slot_acc/slot_turn # +0.0001)
    #slot_acc = sum(p == t for p, t in zip(pred_slots, true_slots)) / len(pred_slots)
    """
    return domain_acc, (P_q, R_q, F1_q), (P_api, R_api, F1_api), (P_total, R_total, F1_total), (P_total_e, R_total_e, F1_total_e)

def evaluate_ret(passage_model, context_model, dev_dataloader, global_index, cfg):
    context_model.eval()
    preds = []
    trues = []
    data_dict = json.loads(open('data/dev_final_processed.json', 'r').read()) # 'data/test_final_processed.json'
    global_kb = json.loads(open('kb/ret_qa.json', 'r').read())
    #data_dict = json.loads(open('data/test.json', 'r').read())
    wrong_cases = []
    correct_cases = []
    total_case = 0
    correct_case1 = 0
    correct_case5 = 0
    correct_case20 = 0

    for batch in dev_dataloader:#tqdm(dev_dataloader, desc="Evaluating"):
        with torch.no_grad():
            for key, val in batch.items():
                if type(batch[key]) is list:
                    continue
                batch[key] = batch[key].to(cfg.device[0])
            
            local_index = build_local_index(passage_model, batch['local_kb'], global_index)
                
            logits = context_model(input_ids=batch["input"],attention_mask=batch["attention_mask"], indices=local_index, topk=20)

            for num in range(len(batch["labels"])):
                data_id = batch["id"][num]
                org_data = data_dict[data_id]['log'][batch["turn_num"][num]]
                labels = batch["labels"][num]
                kb = copy.deepcopy(global_kb)
                kb.extend(batch["local_kb"][num])

                predict_index = logits[num][1].cpu().tolist()
                org_data['predict_ret'] = [kb[p] for p in predict_index]

                if labels!=[] and labels!=[-1]:
                    for l in labels:
                        pred = logits[num]
                        total_case += 1
                        top5, top5_pos = torch.topk(pred[0], k=5)
                        top1, top1_pos = torch.topk(pred[0], k=1)
                        top5_indice = pred[1][top5_pos]
                        top1_indice = pred[1][top1_pos]
                        if l in pred[1].cpu().tolist():
                            correct_case20 += 1
                        if l in top5_indice.cpu().tolist():
                            correct_case5 += 1
                        if l in top1_indice.cpu().tolist():
                            correct_case1 += 1
                            correct_cases.append(org_data)
                        else:
                            wrong_cases.append(org_data)
                        #intent_preds = torch.topk(logits, dim=1).cpu().tolist()
            # change this to test R@1,5,20
            # torch.topk(retriever_scores, topk, dim=1)
            """
            probs = F.softmax(logits,dim=-1)
            prob = []
            for i in range(len(intent_preds)):
                prob.append(probs[i,intent_preds[i]].cpu())
            pred = intent_preds
            true = batch["domain"].cpu().tolist()
            preds += pred
            trues += true
            for i in range(len(pred)):
                if pred[i]!=true[i]:
                    turn_id = batch['id'][i]
                    turn_num = batch['turn_num'][i]
                    if len(data_dict[turn_id])>turn_num: # or (turn_num in data_dict[turn_id])
                        turn = data_dict[turn_id][turn_num]
                        wrong_domain.append((turn, pred[i]))
            """

    json.dump(correct_cases, open('correct_results.json', 'w'), indent=2, ensure_ascii=False)
    json.dump(wrong_cases, open('wrong_results.json', 'w'), indent=2, ensure_ascii=False)
    e = 1e-7
    recall_1 = correct_case1/(total_case+e)
    recall_5 = correct_case5/(total_case+e)
    recall_20 = correct_case20/(total_case+e)
    return recall_1, recall_5, recall_20

def test_domain(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_path)#cfg.model_path
    model = Bert_Model(cfg,tokenizer)#,tokenizer
    model.load_state_dict(torch.load(os.path.join(cfg.model_path, "model.pt")))
    model.to(cfg.cuda_device[0])#is list

    encoded_data = read_data(tokenizer)
    data = encoded_data['train']
    data.extend(encoded_data['dev'])
    data.extend(encoded_data['test'])
    train_dataloader=DataLoader(data, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn) 
    test_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn)
    sales,job,uncertain = evaluate(model, train_dataloader, cfg, return_result=True)
    #print(score)
    return 

def train(cfg):
    cfg.exp_path = 'experiments_retrieve'
    cfg.batch_size = 16 # 32
    cfg.lr = 1e-5

    json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
    tokenizer = BertTokenizer.from_pretrained(cfg.api_encoder_path)   
    # Add special tokens
    init_vocab_size=len(tokenizer)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    logging.info('Added special tokens:{}'.format(special_tokens))
    tokenizer.add_special_tokens(special_tokens_dict)
    logging.info('Special token added, vocab size:{}-->{}'.format(init_vocab_size, len(tokenizer)))

    encoded_data = read_data(tokenizer, retrieve=True)
    if cfg.only_one_model:
        model = BertForNextSentencePrediction.from_pretrained('bert-base-chinese') #EncoderModel(cfg,tokenizer)
        model.resize_token_embeddings(len(tokenizer))
        model.to(cfg.device[0])
    else:
        model = EncoderModel(cfg,tokenizer)

    train_dataloader=DataLoader(encoded_data['train'], batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn) 
    dev_dataloader=DataLoader(encoded_data['dev'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn)
    test_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=collate_fn)
   
    optimizer, scheduler = get_optimizers(num_samples=len(encoded_data['train']) ,model=model, lr=cfg.lr)
    global_step = 0
    metrics_to_log = {}
    best_score = -1
    for epoch in range(cfg.epoch_num):
        model.train()
        epoch_loss = 0
        epoch_step = 0
        num_batches = 0

        for batch in tqdm(train_dataloader):
            num_batches += 1

            # Transfer to gpu
            if torch.cuda.is_available():
                for key, val in batch.items():
                    if type(batch[key]) is list:
                        continue
                    batch[key] = batch[key].to(cfg.device[0])
                if cfg.only_one_model:
                    loss = model(input_ids=batch["input"], attention_mask=batch["input_attention"], token_type_ids=batch["input_type"],labels=batch["label"]).loss
                else:
                    loss = model(input_sent=batch["context"], attention_sent=batch["context_attention"], input_triple=batch["triple"], attention_triple=batch["triple_attention"],label=batch["label"])[0]
                loss.backward()
                epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                epoch_step += 1
                if epoch_step % cfg.gradient_accumulation_steps == 0 or num_batches==len(train_dataloader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

        logging.info("Epoch loss: {}".format(epoch_loss / num_batches))
        print(epoch_loss / num_batches)
        # Evaluate and save checkpoint
        score, precision, recall, f1 = evaluate(model, test_dataloader, cfg) # dev_dataloader
        metrics_to_log["eval_score"] = score
        logging.info("score: {}, precision: {}, recall: {}, f1: {}".format(score, precision, recall, f1))
        s = score + recall + f1
        print("score: {}, precision: {}, recall: {}, f1: {}".format(score, precision, recall, f1))
        if s > best_score:
            logging.info("New best results found! Score: {}".format(score))
            #model.bert_model.save_pretrained(cfg.save_dir)
            if cfg.only_one_model:
                if not os.path.exists(cfg.bert_save_path):
                    os.mkdir(cfg.bert_save_path)
                model.save_pretrained(cfg.bert_save_path)
                tokenizer.save_pretrained(cfg.bert_save_path)
            else:
                if not os.path.exists(cfg.context_save_path):
                    os.mkdir(cfg.context_save_path)
                if not os.path.exists(cfg.triple_save_path):
                    os.mkdir(cfg.triple_save_path)
                if not os.path.exists(cfg.retrieval_save_path):
                    os.mkdir(cfg.retrieval_save_path)
                model.sentence_model.save_pretrained(cfg.context_save_path)
                model.triple_model.save_pretrained(cfg.triple_save_path)
                tokenizer.save_pretrained(cfg.context_save_path)
                tokenizer.save_pretrained(cfg.triple_save_path)
                torch.save(model.state_dict(), os.path.join(cfg.retrieval_save_path, "model.pt"))
            best_score = s
    #model.load_state_dict(torch.load(os.path.join(cfg.save_dir, "model.pt")))
    #score = evaluate(model, test_dataloader, cfg)
    #print(score)
    return

def inference(cfg):

    return

if __name__ == "__main__":
    #cfg.debugging = True
    #cfg.train_retrieve = False
    if cfg.exp_path=='':
        #cfg.exp_name = 'retrieve'
        experiments_path = './experiments_retrieve'
        cfg.exp_name = 'retrieve_api'
        cfg.exp_path = os.path.join(experiments_path, cfg.exp_name)
        if not os.path.exists(cfg.exp_path):
            os.mkdir(cfg.exp_path)
    cfg.mode = 'train_retrieve'
    cfg._init_logging_handler()

    # train_api_model(cfg)
    train_api_model_retrieve(cfg)
    # test_api_model_retrieve(cfg)

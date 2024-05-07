from fileinput import filename
import json
import os
from pydoc import doc
import re
import pandas as pd
import numpy as np
import random
import queue
#import datasets
import copy
import re
from transformers import BertTokenizer
from sentence_transformers import models, SentenceTransformer, losses, datasets, util
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
#tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def remove_noisy_utt(utt): 
    toxics = ['人工MM正在帮您核实问题，请您稍等~', 
    '(*^__^*)','o（*￣▽￣*）o','\t'
    '[em_40]','[em_48]',
    '[em_49]','[em_32]',
    '[em_34]','[em_36]',
    '[em_42]','[em_29]']
     # '季节在变，深情不变，遇见您是最美好的时光！'
    utt_new = utt
    for toxic in toxics:
        utt_new = utt_new.replace(toxic, '')
    if "#" in utt_new:
        pieces = utt_new.split("#")
        new_piece = []
        for piece in pieces:
            if "{" in piece:
                if piece[-1]=="}":
                    try:
                        new_piece.append(json.loads(piece)["orgiMsgContent"])
                    except:
                        new_piece.append(piece)
                else:
                    tmp_piece = piece.split("}")[-1]
                    piece = piece.replace(tmp_piece, "")
                    try:
                        new_piece.append(json.loads(piece)["orgiMsgContent"])
                    except:
                        new_piece.append(piece)
                    new_piece.append(tmp_piece)
            else:
                new_piece.append(piece)
        utt_new = "".join(new_piece)         
    return utt_new

def get_data(): 
    data = pd.read_table("流量对话.csv", delimiter=",", encoding="gbk")
    url =  data.to_dict()['req_url']
    system =  data.to_dict()['坐席']
    user =  data.to_dict()['客户']
    bot =  data.to_dict()['机器人']
    dialogs = []
    dialog = []
    id = ''
    role = ''
    delete_flag = 0
    for i in range(808):
        # new session
        if url[i]!=id:
            id = url[i]
            if dialog!=[]:
                dialogs.append(dialog)
            dialog = []
            turn = {'user':'',
                    'system':'',
                    'api-name':'',
                    'api-params':{},
                    'api-results':''}
        if str(user[i])!='nan':
            if role!='user':
                flag = 1
                role = 'user'
            if flag==1:
                if turn['user']!='' and turn['system']!='':
                    dialog.append(copy.deepcopy(turn))
                    turn = {'user':'',
                            'system':'',
                            'api-name':'',
                            'api-params':{},
                            'api-results':''}
            if "@" not in user[i]:
                if delete_flag!=1:
                    turn['user'] = turn['user'] + ' ' + user[i]
                else:
                    delete_flag = 0
        else:
            if str(system[i])!='nan':
                if role!='system':
                    role = 'system'
                if ("系统提示" not in system[i]) and ("季节在变" not in system[i]) and ("&" not in system[i]): # "系统提示", "&", 请您稍等
                    # toxic expressions, need to be deleted:
                    # 人工MM正在帮您核实问题，请您稍等~
                    # (*^__^*)
                    if '请您稍等' not in system[i]:
                        turn['system'] = turn['system'] + ' ' + system[i]
                    else:
                        delete_flag=1
                #turn['utterance'] = system[i]
            #elif str(bot[i])!='nan':
            #    role = 'bot'
            #    turn['role']='bot'
            #turn['utterance'] = bot[i]
        #dialog.append(turn)
        #else:
        #    dialog
    dialogs.append(dialog)
    json.dump(dialogs, open("data_annotated.json", 'w'), indent=2, ensure_ascii=False)
    return

def get_qa(): 
    data = pd.read_table("定向流量相关QA.csv", delimiter=",", encoding="gbk")
    questions =  data.to_dict()['标准问题']
    answers =  data.to_dict()['标准答案']
    qa = {}
    for i in range(34):
        qa[questions[i]] = answers[i]
    json.dump(qa, open("qa.json", 'w'), indent=2, ensure_ascii=False)
    return

def get_clustering(sentences):
    model = SentenceTransformer('uer/sbert-base-chinese-nli')#'uer/sbert-base-chinese-nli'
    embeddings = model.encode(sentences)
    corpus_embeddings =embeddings /  np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None, affinity='cosine', linkage='average', distance_threshold=0.1)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
        clustered_sentences[cluster_id].append(sentences[sentence_id])

    return clustered_sentences
""""
# discard because of efficiency
def get_embedding(sentences):
    model = SentenceTransformer('uer/sbert-base-chinese-nli')
    #paraphrases = util.paraphrase_mining(model, sentences)
    embeddings = model.encode(sentences)
    return embeddings
"""
def cal_sim(sentence1, sentence2):
    model = SentenceTransformer('uer/sbert-base-chinese-nli')
    """
    bert = models.Transformer('microsoft/mpnet-base')
    pooler = models.Pooling(
        bert.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    model = SentenceTransformer(modules=[bert, pooler])
    training a sentence model
    loss = losses.MultipleNegativesRankingLoss(model)
    epochs = 1
    warmup_steps = int(len(loader) * epochs * 0.1)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path='mpnet-mnr-squad2',
        show_progress_bar=True
    )
    batch_size = 24
    loader = datasets.NoDuplicatesDataLoader(
        train, batch_size=batch_size
    )
    """

    return
                
def test_preprocess(): 
    logs = json.load(open('jiangsu_directional_flow_data_202306.json', 'r'))
    count = 0
    data = {}
    for ids, dialog in logs.items():
        count = count + 1
        if count < 8000:
            for turn in dialog:
                if 'agent' in turn:
                    turn['agent'] = remove_noisy_utt(turn['agent'])
            data[ids] = dialog
    #json.dump(data, open("test_preprocess.json", 'w'), indent=2, ensure_ascii=False)
    return

def postprocess(): 
    files = ['data_to_be_annotated1.csv',
            'data_to_be_annotated2.csv',
            'data_to_be_annotatedpre1.csv',
            'data_to_be_annotatedpre2.csv']
    for file in files:
        data = pd.read_table(file, delimiter=",")#, encoding="gbk")
        ids =  data.to_dict()['对话id']
        users =  data.to_dict()['用户']
        agents =  data.to_dict()['客服']
        apis =  data.to_dict()['API查询']
        results =  data.to_dict()['返回结果']
        csv_file = open("new_" + file , 'w')
        csv_file.writelines('对话id,用户,客服,API查询,返回结果,\n')
        for num,id in ids.items():
            user = users[num]
            agent = agents[num]
            api = apis[num]
            result = results[num]
            if isinstance(id, str) and isinstance(user, str) and isinstance(agent, str):
                new_agent = remove_noisy_utt(agent.replace('，',','))
                csv_file.writelines(','.join([id, user, new_agent.replace(',','，'),
                api if isinstance(api, str) else '',
                result if isinstance(result, str) else '']) + ',\n')
        csv_file.close()
    #json.dump(data, open("test_preprocess.json", 'w'), indent=2, ensure_ascii=False)
    return

def find_qa_from_logs(): 
    logs = json.load(open('jiangsu_directional_flow_data_202306.json', 'r'))
    freq_resp = {}
    freq_response = {}
    resps = []
    for _, dialog in logs.items():
        for turn in dialog:
            if 'agent' in turn:
                agent = remove_noisy_utt(turn['agent'])
                if agent not in freq_resp:
                    freq_resp[agent] = 0
                    resps.append(agent)
                freq_resp[agent] += 1
    for utt in freq_resp:
        if freq_resp[utt]>2 and len(utt)>30:
            freq_response[utt] = freq_resp[utt]
    c = get_clustering(resps)
    for _, turns in c.items():
        flag = all((tmp not in freq_response) for tmp in turns)
        if (len(turns)>4) and (len(turns[0])>30) and flag:
            freq_response[turns[0]] = len(turns)  
    qa = {}
    for _, dialog in logs.items():
        hist = ''
        for turn in dialog:
            hist = []
            if 'agent' in turn:
                hist.append(turn['user'])
                if len(hist)>5:
                    hist = hist[2:]
                agent = remove_noisy_utt(turn['agent'])
                if (agent in freq_response) or len(agent)>200:
                    qa[''.join(hist)] = agent
                hist.append(turn['agent'])
    json.dump(qa, open("qa_found.json", 'w'), indent=2, ensure_ascii=False)
    return

def retrieve_from_qa(): 
    # get qa database from multiple sources
    model = SentenceTransformer('uer/sbert-base-chinese-nli')
    qa={}
    resp = []
    qa_org = json.load(open('qa_found.json', 'r'))
    pre_qa = json.load(open('qa.json', 'r'))
    for q,a in qa_org.items():
        pre_qa[q] = a
    for q,a in pre_qa.items():
        if a not in resp:
            resp.append(a)
    for q,a in pre_qa.items():
        qa[q] = (a, model.encode([a])[0])
        if a not in resp:
            resp.append(a)
    """
    for q, a in qa.items():
        # get the indexing
        #remove = answers[i][re.match("点", answers[i]).span()[0]:re.match("点", answers[i]).span()[1]]
        piece = a.split('<')
        new_a = ''.join(p.split('>')[-1] for p in piece)
        #new_a=new_a.replace('【','[')
        # the triple contains query for testing; query for preprocessing; answer that are used to replace
        qa[q] = (q, new_a, a)
    """
    # perform matching
    logs = json.load(open('jiangsu_directional_flow_data_202306.json', 'r')) 
    count = 0
    # remove ids that has already been annotated
    
    ids = [i for i in range(4855)]
    random.shuffle(ids)
    ids = ids[:1500]
    log0 = {}
    log1 = {}
    log2 = {}
    log3 = {}
    for org_id, dialog in tqdm(logs.items()):
        if org_id not in ann_ids:
            count = count + 1
            if count in ids:
                hist = ''
                for turn in dialog:
                    hist = hist + turn['user']
                    # selecting turns that are suitable for qa are important
                    if 'agent' in turn:
                        agent = remove_noisy_utt(turn['agent'])
                        emb = model.encode([agent])[0]
                        for q,a in qa.items():
                            # use a[1] to evaluate similarity
                            sim = util.pytorch_cos_sim(emb, a[1]).item()
                            if (a[0] in turn['agent']) or (sim>0.95): #or (turn['agent'] in a[0]):
                                turn['api_name'] = '[QA]'
                                turn['agent'] = remove_noisy_utt(turn['agent'])
                                #turn['result_json'] = a[0]
                                break
                hist = hist + agent
            if count in ids[:750]:
                log0[org_id] = dialog
            elif count in ids[:1500]:
                log1[org_id] = dialog
    convert_json_to_csv(log0, "0")
    convert_json_to_csv(log1, "1")
    #json.dump(logs[], open("clean_dialogspre_annotated.json", 'w'), indent=2, ensure_ascii=False)
    return


def convert_json_to_csv(data, file): 
    #data = json.load(open('clean_dialogspre_annotated.json', 'r'))
    csv_file = open("data_to_be_annotated_day3"+ file + ".csv", 'w')
    csv_file.writelines('对话id,用户,客服,API查询,返回结果,\n')
    for id,dialog in data.items():
        for turn in dialog:
            if 'agent' in turn:
                csv_file.writelines(','.join([id, turn['user'].replace(',','，'),
                turn['agent'].replace(',','，'), turn['api_name'], turn['result_json'].replace(',','，')]) + ',\n')
    csv_file.close()
    return

def get_analysis(): 
    logs = json.load(open('jiangsu_directional_flow_data_202306.json', 'r'))
    ids = []
    count = 0
    for org_id, dialog in logs.items():
        count = count + len(dialog)
        if org_id[:8] not in ids:
            ids.append(org_id[:8])
    print(count)
    print(ids)

def check_repitition(): 
    files = ['data_to_be_annotated1.csv',
            'data_to_be_annotated2.csv',
            'data_to_be_annotatedpre1.csv',
            'data_to_be_annotatedpre2.csv']
    ann_ids = []
    new_ids = []
    for file in files:
        data = pd.read_table(file, delimiter=",")#, encoding="gbk")
        ids =  data.to_dict()['对话id']
        for num, id in ids.items():
            ann_ids.append(id)
    new_files = ['data_to_be_annotated_day30.csv',
            'data_to_be_annotated_day31.csv']
    ann_id = set(ann_ids)
    for file in new_files:
        data = pd.read_table(file, delimiter=",")#, encoding="gbk")
        ids =  data.to_dict()['对话id']
        for num, id in ids.items():
            new_ids.append(id)
    new_id = set(new_ids)
    print(new_id&ann_id)
    return

if __name__ == "__main__":
    #get_data()
    #get_qa()
    #find_qa_from_logs()
    #retrieve_from_qa()
    check_repitition()
    #get_analysis()
    #test_preprocess()
    #postprocess()
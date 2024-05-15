"""
Copyright 2024 Tsinghua University
Author: Yucheng Cai (cyc22@mails.tsinghua.edu.cn)
"""

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

from reader import clean_intent, clean_utt, get_spoken

from get_data import remove_noisy_utt
#tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# '对话id,用户,客服,API查询,返回结果,\n'
def process_annotated_data(): 
    dir = '标注数据'
    files = os.listdir(dir)
    """
    ['标注数据/8月1日任务2-数据1组.xlsx',
    '标注数据/8月2日标注任务2-数据1组.xlsx',
    '标注数据/标注任务2-试标_2组.xlsx',
    '标注数据/标注任务2-试标1组.xlsx',
    '标注数据/标注任务2-数据2（8月1日李沙组）.xlsx',
    '标注数据/标注任务2-数据2（8月2日李沙组）.xlsx']
    """
    all_data = {}
    multi_query_turns = []
    strange_dialog = []
    no_head_file = ['8月1日任务2-数据1组1.csv',
    '8月1日任务2-数据1组2.csv',
    '8月1日任务2-数据1组3.csv']
    all_ids = []
    strange_ids = []
    for file in files:
        file_ids = []
        if 'csv' in file:
            tmp_id = ''
            dialog = []
            if file not in no_head_file:
                data = pd.read_table(os.path.join(dir, file), delimiter=",", encoding="gbk")#, delimiter=",", encoding="gbk")
                ids =  data.to_dict()['对话id']
                users =  data.to_dict()['用户']
                systems =  data.to_dict()['客服']
                api_querys=  data.to_dict()['API查询']
                api_results = data.to_dict()['返回结果']
            else:
                data = pd.read_table(os.path.join(dir, file), delimiter=",", encoding="gbk", header=None)
                ids =  data.to_dict()[0]
                users =  data.to_dict()[1]
                systems =  data.to_dict()[2]
                api_querys=  data.to_dict()[3]
                api_results = data.to_dict()[4]
                #strange = data.to_dict()[5]
            print(f"file:{file},id:{len(ids.keys())},users:{len(ids.keys())},id:{len(ids.keys())},id:{len(ids.keys())}")
            for num,id in ids.items():
                if isinstance(id, str) and len(id)==25:
                    turn = {}
                    turn['user'] = users[num]
                    turn['system'] = systems[num]
                    if isinstance(api_querys[num], str):
                        turn['api_query'] = api_querys[num]
                        if turn['api_query']=='[QA]':
                            turn['api_result'] = ''
                        else:    
                            if isinstance(api_results[num], str):
                                turn['api_result'] = api_results[num]
                            else:
                                turn['api_result'] = ''
                        if ',' in 'api_query':
                            multi_query_turns.append(turn)   
                    else:
                        turn['api_query'] = ''
                        turn['api_result'] = ''
                    if id!=tmp_id:
                        if id not in file_ids:
                            file_ids.append(id) 
                        if dialog!=[]:
                            if len(dialog)>100:
                                print(f"file:{file}, id :{tmp_id}")
                            if tmp_id not in all_data:
                                all_data[tmp_id] = dialog
                            elif tmp_id not in all_ids:
                                all_data[tmp_id].extend(dialog)
                                strange_dialog.append(dialog)
                        tmp_id = id
                        dialog = []
                    if isinstance(turn['user'], str) and isinstance(turn['system'], str):
                        dialog.append(turn)
                elif isinstance(id, str):
                    strange_ids.append(turn)
        all_ids.extend(file_ids)
    json.dump(all_data, open("nl2api_raw.json", 'w'), indent=2, ensure_ascii=False)
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
                if turn['api_query'] not in all_apis:
                    all_apis[turn['api_query']] = 0 
                all_apis[turn['api_query']] = all_apis[turn['api_query']] + 1
                if (turn['api_result']=='' and (turn['api_query'] not in avoid)) or len(turn['api_result'])>100:
                    special_api.append(turn)
    api = {i:all_apis[i] for i in sorted(all_apis, key=lambda x:all_apis[x], reverse=True)}
    print(api)
    #json.dump(special_api, open("check_turn.json", 'w'), indent=2, ensure_ascii=False)
    return

def get_splitted_data(): 
    # 2753 dialogs in total, split 7:1.5:1.5 as train,dev and test set(1927,413,413)
    all_data = json.load(open('nl2api_cleaned_v3.json', 'r'))
    nums = [i for i in range(2753)]
    random.shuffle(nums)
    num = 0
    train = {}
    dev = {}
    test = {}
    for id, dial in all_data.items():
        if nums[num]<1927:
            train[id] = dial
        elif nums[num]<2340:
            dev[id] = dial
        else:
            test[id] = dial
        num += 1
    json.dump(train, open("data/train.json", 'w'), indent=2, ensure_ascii=False)
    json.dump(dev, open("data/dev.json", 'w'), indent=2, ensure_ascii=False)
    json.dump(test, open("data/test.json", 'w'), indent=2, ensure_ascii=False)
    return

def delex(utt, intent): 
    # **$%yinyong^&**
    # double mark that quote turns before
    # （涉敏话术）
    # 月租 no need for delex
    if intent=="查询超出流量" and "【流量提醒】" in turn["system"]:
        pieces = turn["api_result"].split('，')
        # 发生时间：，超出金额：，超出流量
        return
#return value split by “；”

def clean_ann(): 
    original_dial = json.load(open('nl2api_raw.json', 'r'))
    key_words = ['【流量提醒】']
    new_dial = {}
    for id, dial in original_dial.items():
        for turn in dial:
            tmp_pieces = []
            if '【流量提醒】' in turn['system']:
                turn["api_query"] = '查询超出流量'
                pieces = turn['system'].split('【流量提醒】')
                for piece in pieces:
                    if '【中国移动】' in piece:
                        tmp_piece =  '【流量提醒】' + piece.split('【中国移动】')[0] + '【中国移动】' 
                        tmp_pieces.append(tmp_piece)
                turn["api_result"] = '，'.join(tmp_pieces) 
                if ('\t' in turn['system']) or ('10086 ' in turn['system']) or ('10086\t' in turn['system']):
                    turn['system'] = turn['system'].replace('10086\t', '').replace('\t', '').replace('10086 ', '')
                    turn["api_result"] = turn["api_result"] + '，' + '发生时间：'
        new_dial[id] = dial
    json.dump(new_dial, open("nl2api_cleaned.json", 'w'), indent=2, ensure_ascii=False)
    return

def clean_qa(): 
    original_qa = json.load(open('qa_found.json', 'r'))
    dirty_words = ['【流量提醒】']
    new_qa = {}
    for q, a in original_qa.items():
        flag = 0
        for dirt in dirty_words:
            if dirt in a:
                flag = 1
                break
        if flag == 0:
            new_qa[q] = a   
    json.dump(new_qa, open("qa_cleaned.json", 'w'), indent=2, ensure_ascii=False)
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

def analysis_result():
    reserved_keys = ['用户', '查询结果', '查询', '查询-生成', '查询结果-生成'
        , '客服', '客服-生成', '客服-检索']
    all_apis = {}
    result = json.load(open('experiments/medium4/best_model/result.json', 'r'))
    # experiments/large3/best_model/result.json
    qa = []
    api = []
    response = []
    qa_num = 0
    for dial in result:
        context = ''
        gt_context = ''
        for turn in dial:
            context = context + turn['用户'] + turn['客服-生成']
            gt_context = gt_context + turn['用户'] + turn['客服']
            if turn['查询']!=turn['查询-生成'] or turn['bleu'] < 10.0:
                new_turn = {}
                new_turn['context'] = context
                new_turn['gt_context'] = gt_context
                for k in reserved_keys:
                    if k in turn:
                        new_turn[k] = turn[k]
                if (turn['查询'] =='[QA]' or turn['查询-生成'] =='[QA]') and turn['查询']!=turn['查询-生成']:
                    qa_num += 1
                    qa.append(new_turn)
                elif turn['bleu'] < 10.0 and turn['查询']==turn['查询-生成']:
                    #qa_num += 1
                    response.append(new_turn)
                else:
                    #api_num += 1
                    api.append(new_turn)

    json.dump(qa, open("analysis/bad_qa1.json", 'w'), indent=2, ensure_ascii=False)
    json.dump(api, open("analysis/bad_api1.json", 'w'), indent=2, ensure_ascii=False)
    json.dump(response, open("analysis/bad_turns1.json", 'w'), indent=2, ensure_ascii=False)
    return

def split_api_result(api_result):
    splitted_results = []
    if '，业务名称：' in api_result:
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

def postprocess_data():
    import re
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_raner_named-entity-recognition_chinese-large-generic')
    
    data_path_train = 'data/train.json'
    data_path_dev = 'data/dev.json'
    data_path_test = 'data/test.json'
    train_data = json.loads(open(data_path_train, 'r').read())
    dev_data = json.loads(open(data_path_dev, 'r').read())
    test_data = json.loads(open(data_path_test, 'r').read())
    source_data = {'train':train_data,'dev':dev_data,'test':test_data}
    global_kb = []
    start_delex = '[开始时间]'
    end_delex = '[结束时间]'
    phone_delex = '[手机号]'
    id_delex = '[身份证号]'
    number_delex = '[数字]'
    for file in source_data:
        count = 0
        dials = source_data[file]
        new_dials = {}
        for id, dial in tqdm(dials.items()):
            new_dial = {'log': dial}
            local_kb = []
            hist = ['']
            for turn_num in range(len(dial)):
                turn = dial[turn_num]
                case = {}
                case['id'] = id
                case['turn_num'] = turn_num
                api_query = clean_intent(turn['api_query'])
                
                turn['user'], turn['system'], turn['api_result'] = clean_sensitive_model(ner_pipeline, turn)

                turn['user'] = re.sub(r'\d{18}', id_delex, turn['user'])
                turn['system'] = re.sub(r'\d{18}', id_delex, turn['system'])

                turn['user'] = re.sub(r'\d{11}', phone_delex, turn['user'])
                turn['system'] = re.sub(r'\d{11}', phone_delex, turn['system'])

                turn['user'] = re.sub(r'\d{6}', number_delex, turn['user'])
                turn['system'] = re.sub(r'\d{6}', number_delex, turn['system'])

                if turn['api_result']!="":
                    if '开始时间' in turn['api_result']:
                        splitted = turn['api_result'].split('开始时间：')
                        for piece in splitted[1:]:
                            start_time = piece.split('，')[0]
                            if start_time!='':
                                turn['api_result'] = turn['api_result'].replace(start_time, start_delex)
                                turn['system'] = turn['system'].replace(start_time, start_delex)
                    
                    if '结束时间' in turn['api_result']:
                        splitted = turn['api_result'].split('结束时间：')
                        for piece in splitted[1:]:
                            end_time = piece.split('，')[0]
                            if end_time!='':
                                turn['api_result'] = turn['api_result'].replace(end_time, end_delex)
                                turn['system'] = turn['system'].replace(end_time, end_delex)

                    if '申请时间' in turn['api_result']:
                        splitted = turn['api_result'].split('申请时间：')
                        for piece in splitted[1:]:
                            start_time = piece.split('，')[0]
                            if start_time!='':
                                turn['api_result'] = turn['api_result'].replace(start_time, start_delex)
                                turn['system'] = turn['system'].replace(start_time, start_delex)
                
                    if '办理时间' in turn['api_result']:
                        splitted = turn['api_result'].split('办理时间：')
                        for piece in splitted[1:]:
                            start_time = piece.split('，')[0]
                            if start_time!='':
                                turn['api_result'] = turn['api_result'].replace(start_time, start_delex)
                                turn['system'] = turn['system'].replace(start_time, start_delex)

                    if ('时间' in turn['api_result'] and ('优惠时间' not in turn['api_result']) and ('开放时间' not in turn['api_result']) and ('生效时间' not in turn['api_result']) and ('营业时间' not in turn['api_result']) and ('结束时间' not in turn['api_result']) and ('业务时间' not in turn['api_result'])):
                        splitted = turn['api_result'].split('时间：')
                        for piece in splitted[1:]:
                            start_time = piece.split('，')[0]
                            if start_time!='':
                                turn['api_result'] = turn['api_result'].replace(start_time, start_delex)
                                turn['system'] = turn['system'].replace(start_time, start_delex)
                    
                    turn['api_result'] = re.sub(r'\d{18}', id_delex, turn['api_result'])
                    turn['api_result'] = re.sub(r'\d{11}', phone_delex, turn['api_result'])
                    turn['api_result'] = re.sub(r'\d{6}', number_delex, turn['api_result'])


                if '查询用户已办理的业务' in api_query or api_query=='查询流量信息' or api_query=='查询本月月租':
                    if turn['api_result']!="":
                        if '，业务名称：' not in turn['api_result']:
                            local_kb.append(turn['api_result'])
                        else:
                            splitted = turn['api_result'].split('，业务名称：')
                            local_kb.append(splitted[0])
                            for piece in splitted[1:]:
                                local_kb.append(('业务名称：'+piece))

                elif api_query!='' and api_query!='[QA]':
                    if turn['api_result']!="":
                        if '，业务名称：' not in turn['api_result']:
                            if turn['api_result'] not in global_kb:
                                global_kb.append(turn['api_result'])
                        else:
                            splitted = turn['api_result'].split('，业务名称：')
                            if splitted[0] not in global_kb:
                                global_kb.append(splitted[0])
                            for piece in splitted[1:]:
                                append_piece = '业务名称：'+piece
                                if append_piece not in global_kb:
                                    global_kb.append(append_piece)

            new_dial['local_kb'] = local_kb
            if local_kb != []:
                count += 1
            new_dials[id] = new_dial
        print(f"file:{file},total_session:{len(dials.keys())},local_kb_num: {count}")
        json.dump(new_dials, open('data/' + file + '_processed.json', 'w'), indent=2, ensure_ascii=False) 
    json.dump(global_kb, open('data/global_kb.json', 'w'), indent=2, ensure_ascii=False)  
    return

def clean_sensitive_model(ner_pipeline, turn):
    threshold = 0.26
    ignore_list = ['人工', '转人工', '谢谢', '的亲', '有的', '港澳台', '中国残联', '中国残疾人联合', '中国残疾人联合印章', '优惠', 'app', 
    '啊', 'app', '知道的', '吗 亲',  '吗亲', '记得', '亲戚', '左右', '呀 亲', '', ' ', '  ', 
    '通用', '知道了', '知道', '知道', 'g', '流量', '好的', 'qq', ' 亲', '亲']
    delex = {'LOC':'[地名]',
    'GRP':'[组织名]',
    'PER':'[人名]',
    }
    user = turn['user']
    system = turn['system']
    if 'api_result' in turn:
        api_result = turn['api_result']
    else:
        api_result = ''
    if len(user)<300 and len(system)<300 and len(user)>3 and len(system)>3:
        try:
            user_result = ner_pipeline(turn['user'])
            system_result = ner_pipeline(turn['system'])
            for piece in user_result['output']:
                if piece['type']== 'LOC' or piece['type']== 'GRP' or piece['type']== 'PER':
                    if piece['span'] not in ignore_list:
                        user = user.replace(piece['span'], delex[piece['type']], 1)

            for piece in system_result['output']:
                if piece['type']== 'LOC' or piece['type']== 'GRP' or piece['type']== 'PER':
                    if piece['span'] not in ignore_list:
                        system = system.replace(piece['span'], delex[piece['type']], 1)

            #print(user)
            #print(system)
            # {'output': [{'type': 'CORP', 'start': 4, 'end': 13, 'span': '貝塞斯達遊戲工作室'}, {'type': 'CW', 'start': 17, 'end': 20, 'span': '辐射4'}]}
            schema ={'公司名':	'CORP','创作名':'CW',
            '其他组织名':'GRP','地名':'LOC',
            '人名':'PER','消费品':'PROD'}
            if api_result!='' and len(api_result)<300:
                api_processed = ner_pipeline(turn['api_result'])
                for piece in api_processed['output']:
                    if piece['type']== 'LOC' or piece['type']== 'GRP' or piece['type']== 'PER':
                        if piece['span'] not in ignore_list:
                            api_result = api_result.replace(piece['span'], delex[piece['type']], 1)

        except:
            print(f'length exceded,length:{len(system)}')
            
    if 'api_result' in turn:
        return user, system, api_result
    else:
        return user, system

def clean_sensitive(turn):
    user = turn['user']
    system = turn['system']
    
    # if 
    if '开始时间' in turn['api_result']:
        turn 

    # r'\d{11}'

    return user, system

def get_unlabel_data():
    phone_delex = '[手机号]'
    id_delex = '[身份证号]'
    number_delex = '[数字]'
    start_delex = '[开始时间]'
    end_delex = '[结束时间]'
    data_path_train = 'data/train.json'
    data_path_dev = 'data/dev.json'
    data_path_test = 'data/test.json'
    all_data = json.loads(open('pre_ann_data/jiangsu_directional_flow_data_202306.json', 'r').read())

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_raner_named-entity-recognition_chinese-large-generic')

    train_data = json.loads(open(data_path_train, 'r').read())
    dev_data = json.loads(open(data_path_dev, 'r').read())
    test_data = json.loads(open(data_path_test, 'r').read())
    unlab_data = {}
    count = 0
    error_count = 0
    error_count_mix = 0
    error_turns = []
    system_signal = '季节在变，深情不变'
    for id, dial in tqdm(all_data.items()):
        if (id not in train_data) and (id not in dev_data) and (id not in test_data):
            count = count + 1
            user_all = ''.join(t['user'] for t in dial)
            if system_signal not in user_all:
                for turn in dial:
                    turn['user'] = remove_noisy_utt(turn['user'])
                    turn['user'] = clean_utt(turn['user'])
                    if 'agent' in turn :
                        turn['system'] = remove_noisy_utt(turn['agent'])
                        turn['system'] = clean_utt(turn['system'])
                        turn.pop('agent')
                    else:
                        if turn != dial[-1]:
                            turn['system'] = ''
                            error_count = error_count + 1
                            error_turns.append(turn)
                        else:
                            dial.remove(turn)
                            break
                    turn['user'], turn['system'] = clean_sensitive_model(ner_pipeline, turn)

                    turn['user'] = re.sub(r'\d{18}', id_delex, turn['user'])
                    turn['system'] = re.sub(r'\d{18}', id_delex, turn['system'])

                    turn['user'] = re.sub(r'\d{11}', phone_delex, turn['user'])
                    turn['system'] = re.sub(r'\d{11}', phone_delex, turn['system'])

                    turn['user'] = re.sub(r'\d{6}', number_delex, turn['user'])
                    turn['system'] = re.sub(r'\d{6}', number_delex, turn['system'])

                    if '开始时间' in turn['system']:
                        splitted = turn['system'].split('开始时间：')
                        for split in splitted[1:]:
                            start_time = split.split('，')[0]
                            if start_time!='':
                                turn['system'] = turn['system'].replace(start_time, start_delex)
                    
                    if '结束时间' in turn['system']:
                        splitted = turn['system'].split('结束时间：')
                        for split in splitted[1:]:
                            end_time = split.split('，')[0]
                            if end_time!='':
                                turn['system'] = turn['system'].replace(end_time, end_delex)

                    if '申请时间' in turn['system']:
                        splitted = turn['system'].split('申请时间：')
                        for split in splitted[1:]:
                            start_time = split.split('，')[0]
                            if start_time!='':
                                turn['system'] = turn['system'].replace(start_time, start_delex)
                    
                    if '时间' in turn['system']:
                        splitted = turn['system'].split('时间：')
                        for split in splitted[1:]:
                            start_time = split.split('，')[0]
                            if start_time!='':
                                turn['system'] = turn['system'].replace(start_time, start_delex)

                    turn.pop('api_name')
                    turn.pop('result_json')
                unlab_data[id] = dial
            else:
                error_count_mix += 1
    print(f"mix_error:{error_count_mix}")
    print(error_count)
    print(error_turns)
    print(count)
    json.dump(unlab_data, open('data/unlabel.json', 'w'), indent=2, ensure_ascii=False)  
    return 

def clean_sensitive_manual():
    keywords_unlabel = pd.read_table("流量对话.csv", delimiter=",", encoding="gbk")
    return


def get_sample_data():
    data_path_dev = 'data/dev_processed.json'
    dev_data = json.loads(open(data_path_dev, 'r').read())
    tmp = []
    for id, dial in dev_data.items():
        tmp.append(dial)
    random.shuffle(tmp)
    json.dump(tmp[:100], open("data/sample.json", 'w'), indent=2, ensure_ascii=False)
    return

def clean_bad_utt():
    filename = 'data/train_processed.json'#'data/dev_processed.json', 'data/test_processed.json'
    new_filename = 'data/train_processed_cleaned.json'#'data/dev_processed_cleaned.json', 'data/test_processed_cleaned.json'
    company_utterances = ['江苏[组织名]', '你们移动', '江苏移动','垃圾移动', '苏州集团地铁', '中国移动', '移动', '电信', '联通']
    replace_utterances = ['txt', '**$%', '^&**', '（*^▽^*）', '(*^▽^*)',
     'o(*￣-￣*)o', '\t', '？？', '? ? ?', '~', '（*^__^*）', '\\n', '*^__^*',
     '亲，请问还在吗？', '长时间不回复 系统会自动掉线哦', '（?￣ ? ￣?）', '“',
      '”', 'online service', '亲，请问还在线吗？', '---', '--', 'o（*￣-￣*）o',
      '（温馨提示：您可以通过稍后下发的短信链接查询问题解决进展哦）',
      '-changjingzhi进人工入口', '人工changjingzhi']
    bad_utterances = ['涉敏话术',  '不是坑人吗', '牛逼', '你们移动真的够了，又贵东西又少','一声不坑就扣了',
      '你们偷偷给我办流量套餐都不知道扣了我多少钱了','你们太坑了','你们这不坑人吗','太恶心人了','扯淡呢',
      '我是脑子有病还是啥','我本家就讨厌你们这行吗捆绑关系','也太恶心了吧','你们这就是强制消费12345去投诉。必须退我钱！！！','你们就是强制消费啊',
      '我是脑子有病还是啥','那也坑了我三十四啊','也太恶心了吧','你们这就是强制消费12345去投诉。必须退我钱！！！','你们就是强制消费啊',
      '用个鬼啊','改了个什么玩意','我特么都充多少钱流量了','打12315投诉了','你们就是强制消费啊', '錢，最好乖乖給我退', 
      '那你們就繼續坑客戶的錢吧。要不然也沒有收入啊','你們也繼續讓客戶用唄。你們怎麼想的那麽精明', '錢，最好乖乖給我退',
      '什麼都你們說了算。憑什麼啊',  '所以就可以仁你們為所欲為是吧。霸王條款是吧。',
      '这个太坑了。天天扣我钱', '所以就可以仁你們為所欲為是吧。霸王條款是吧。','恶心', '你们涉嫌虚假营销。再不解决我去[组织名]投诉',
      '乱扣费', '抢劫', '投诉', '中国移动是真牛。我可以报警吗。', '毫无卵用', '工信部', '鬼', #'违约',
      '过分', '你们工作效率那么慢吗', '傻逼', '看不懂中国话？', '不人性化', '不合理', '坑', '半天憋不出一句话',
      '天价移动', '太乱了', '不想做冤大头了', '歧视我们吗', '磨磨唧唧', '诱导消费', '套路', '驴唇不对马嘴',
      '滚', '尼玛', '滚蛋', '磨什么都解决不了', '冤枉', '垃圾', '跟宣传不符',
      '骗', '黑店', '想钱想疯了', '什么玩意', '磨什么都解决不了', '差劲', '乱收费', '虚假宣传',
      '不要在着给我逼大呼', '脑残', '私自给我开了一年的套餐', '薅老客户羊毛', '磨什么都解决不了', '霸王条款', '私自扣费',
      '垄断', '流氓', '差劲', '12315投诉', '折腾消费者', '你妈的', '老子', '劳资',
      '抢', '欺诈', '早日倒闭', '12315举报', '逼逼', '想钱想疯了', '怎么不去抢',
      '毛', '天价收费', '工信部', '栓Q', '乱计流量', '什么东西', '真的无语', '好离谱',
      '乱扣', '强制绑定', '你们这边怎么回事', '我脑壳有包办这种套餐', '瞎扯', '过分', '没经过我的同意', '人傻钱多',
      '乱搞', '不合理扣费', '恐怖', '倒霉', '老百姓都消费不起了', '死', '没经过我的同意', '故意', '强制', 
      '毒', '羊毛', '吃相太难看了', '黑心', '睁眼说瞎话', '我炒', '抢钱', '智障', '特么', '跟我这打太极',
      '瞎', '文字游戏', '离谱', '你是业务不熟吗', '踢皮球', '可怕', '搞笑', '特么', '跟我这打太极',
      '流量超过没有短信提醒', '脑子有问题', '离谱', '你是业务不熟吗', '冤种', '不要扯那么多', '乱开', '套路', '效率太低',
      '乱七八糟', '脑子有问题', '投诉', '你是业务不熟吗', '有病', '差评', 'TM', 'tm', '晕', '忽悠', '人机', '玛德',
      '玩意', '神经病', '骚扰', 'mlgb', '泄露', '体验感很差', '冤大头', '屁', '他妈', '疯', '宰', '歧视']
    time_reg = ['\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '\d{4}-\d{2}-\d{2}', '\d{2}:\d{2}:\d{2}', '\d{4}-\d{2}', '\d{3}-\d{2}',
     '\d{4}年\d{1,2}月\d{1,2}日', '\d{4}年\d{1,2}月', '\d{4}年\d{2}月', '\d{1,2}日\d{1,2}时\d{1,2}分']
    time_utterances = ['5月8日12时7分', '2024/6/11', '2024.4.1']
    place_utterances = ['市晋陵中路 337 号。天宁区 常州天宁北直街社区中心店天宁区自办厅省内跨区销户，省际跨区销户 8:00-18:00 暂无北直街 26 号。天宁区 常州郑陆中心店天宁区自办厅携号转网，宽带销户，省内跨区补卡，省内跨区销户，省际跨区补卡，省际跨区销户，定制终端销售，NFC-SIM 卡，吉祥号码 8:00-17:[手机号]63(只接受携转咨询) 常州市郑陆镇迎宾路 6号。钟楼区 常州邮电路中心店钟楼区自办厅携号转网，宽带销户，省际跨区补卡，省内跨区销户，省际跨区销户 9:00-18:[手机号]89(只接受携转咨询)常州市钟楼区邮电路 12 号。钟楼区 常州钟楼大名城中心店钟楼区自办厅省际跨区补卡,省内跨区销户,省际跨区销户 9:00-11:30,14:00-17:00 暂无常州市钟楼区大名城 40 幢 19 号、20 号。钟楼区 常州钟楼开发区中心店钟楼区自办厅携号转网，宽带销户，省内跨区补卡，省际跨区补卡，省内跨区销户，省际跨区销户 9:00-11:30，13:30-17:00 暂无常州市钟楼区玉龙路东侧科技街 C座一层商铺(麦当劳隔',
    '江苏 苏州 吴中区 长桥分局太湖东路营业厅】【营业状态】正常营业 【具体地址】苏州市吴中区太湖东路 99号2号【营业时间】8:30-17:30 【服务电话】0512-[数字159。【江苏 苏州 吴中区 高新区分局东吴广场营业厅【营业状态】正常营业【具体地址】吴中区长桥街道东吴北路 170号【营业时间】8:30-17:00【服务电话】[手机号1。【江苏 苏州 吴中区 城南分局石湖东路营业厅】【营业状态】正常营业【具体地址】吴中城南街道石湖东路 173 号新盛花园6幢 101 室【营业时间】8:30-19:00【服务电话】[手机号]',
    '江苏 [地名] 广益佳润大厦标杆厅[营业状态] 正常营业[具体地址][地名]广南路388号1楼F105[营业时间]9:00-18:00',
    '[繁忙厅 [地名] 苏州 常熟市 琴川分局海虞北路 64 号营业厅][营业状态] 正常营业 [具体地址][地名]苏州市常熟市海虞北路 64 号 [营业时间]8:30-17:00[服务电话] 0512-[数字]00。这家您看下近嘛',
    '【江苏[地名][地名]盐城盐都城南立信指定专营店】【营业状态】正常营业 【具体地址】[地名][地名]福裕路 17 号 256 幢 kao 近[组织名]【营业时间】8:00-12:00;14:00-18:00',
    '江苏 苏州 昆山市 亭林分局前进西路营业厅，营业状态: 正常营业，具体地址:[地名]1390 号，营业时间: 8:00-18:00，服务电话:0512-[数字]03',
    '江苏 苏州 常熟市 琴川分局海虞北路64号', '苏州市常熟市海虞北路64号', '桂吴村', '惠山区水厂路61号，无锡市惠山区玉祁周忱路91号',
    '睢宁县文学路8号（文学广场对面）', '江苏 徐州 睢宁县 文学路', '常州天宁凤凰路中心店天宁区', '雕庄菱溪名居1栋7号（一楼）',
    '江苏 常州 天宁区 常州郑陆中心店天宁区', '迎宾路6号', '诺诚高第委托加盟店天宁区', '常州诺诚高第商务广场2幢112号', '澄张路山观江南御园',
    '江南御园', '仪征市 西园路沟通100店', '仪征西园路88号', '广陵区文昌中路548号', '承德路回迁楼', 
    '江苏省[地名]118号', '江宁区竹翠路50号', '江宁陶吴狮山路50号', '江宁区天元东路865号天景山商业中心8幢144号', '金陵新四村27号', 
    '南京市鼓楼区中山北路301号', '南京市鼓楼区定淮门大街7号', '[地名]150号', '新狮新苑62幢02号一楼', '青山路5号',
    '华通路58号新天地商业街1幢17室', '新民洲民心家园A区66栋66-05号', '镇江市京口区恒美嘉园2A幢第1层109室', '镇江市我家山水瑞雪苑6幢第1层104室', '镇江市学府路88-8号大观天下小区11幢106室',
    '象山花园一期1幢105-2', '越河街37号', '镇江京口区学府路301号', '镇江市中山东路338号', '京口区学府路301号', '洪江路69号一楼',
    '南通城市职业技术学院', '苏州工业园区', '安徽合肥', '文星广场', '[地名]283号', '苏州大道东283号', '江苏 南京 江宁区', '苏州市吴江区教育局',
    '庐山路226号', '南京市建邺区庐山路226号', '建邺区 庐山路营业厅', '建邺区油坊桥', '四川', '江宁区天元东路865号天景山商业中心8幢144号', '江阴璜土小湖路180号',
    '建邺区 高庙路营业厅', '建邺区 云河路营业厅', '南京市建邺区云河路33号', '建邺区 兴隆大街营业厅', '南京市建邺区河西兴隆大街200-7号', '江阴市山观东街75号', '江阴月城团结路55号',
    '苏州玉山路营业厅', '江阴澄南路338-340号', '徐州淮海西路111号移动大楼', '江苏 徐州 泉山区 淮海路营业厅', '江苏 徐州 泉山区 奎园营业厅', '春月园3号楼1楼门面房', '矿山路32号',
    '江苏 徐州 泉山区 矿山路营业厅', '4号楼1单元1602', '六合区 文晖路雨庭花园营业厅', '六合雄州文晖路105、107号', '六合区 六合龙池龙湾路营业厅', '六合龙池龙湾路84号', '六合区 六合冶山街道营业厅', '南京市六合区冶山街道石柱林路40号', '六合区 六合横梁镇营业厅',
    '六合兴镇路166号', '六合区 六合马鞍镇城西营业厅', '南京市六合区城西路515号', '江苏 徐州 鼓楼区 手机卖场徐州淮海路店营业厅鼓楼区自办厅', '徐州市淮海东路125号1楼', '阜宁胜利北路指定专营店', '越溪镇', '南京邮电大学', '阜宁县 阜宁阜城大街沟通 100 店 【具体地址】[地名][地名]阜宁县阜城大街 234 号',
    '南通崇川区', '江苏 南通 崇川区 洪江路旗舰店', '洪江路69号二楼', '江苏 南通 开发区 南通开发区沟通100店', '开发区厦门路11号', '师苑路丰园小区四单元407', '江苏阜宁营业厅', '阜宁县 阜宁阜城大街沟通 100 店', '阜宁县阜城大街 234 号',
    '苏州昆山', '环镇北路', '昆山市前进西路1390号', '康居路59号', '春风南岸小区', '厦门路11号', '广东江门江海品禮樂', '江阴市', '福州', '泰州', '浙江', '栖霞区',
    '江苏 苏州 吴中区 长桥分局太湖东路营业厅】【营业状态】正常营业 【具体地址】苏州市吴中区太湖东路 99号2号【营业时间】8:30-17:30 【服务电话】0512-[数字159。【江苏 苏州 吴中区 高新区分局东吴广场营业厅【营业状态】正常营业【具体地址】吴中区长桥街道东吴北路 170号【营业时间】8:30-17:00【服务电话】[手机号1。【江苏 苏州 吴中区 城南分局石湖东路营业厅】【营业状态】正常营业【具体地址】吴中城南街道石湖东路 173 号新盛花园6幢 101 室【营业时间】8:30-19:00【服务电话】[手机号]', '江苏大学',
    '常州 南夏墅街道', '常州市', '太仓', '宿迁准海技师学院', '淮安市', '[地名1口岸镇通扬路115 号', '上阳路9号', '江阴市', '镇江', '常州', '连云港', '栖霞区',
    '廣東', '盐城', '苏州', '扬州', '四川', '南京', '澳洲', '徐州', '江阴市', '镇江', '常州', '重庆',
    '江苏', '盐城', '苏州', '扬州', '四川', '南京', '澳洲', '徐州', '江阴市', '镇江', '常州', '浙江',
    ]
    number_utterances = ['js31492','StaffID: 31492', '09103W8F74HAF1','StaffID: 31492', 'Gmy[数字]0926', 'js31500',
    'JS22753', '0519[数字]27', '[人名]25357', '23911', '7239',
     '09108W8V76FFT1', '31488', '25357', 'JS31493', '158****5892', '023-06-[数字]-07-01', '025-[数字]80', 'SC18270T[手机号]109']
    number_reg = ['js\d{5}', 'JS\d{5}', '\d{3} \d{4} \d{4}', '\d{5}X', '(?!10086)\d{5}']
    name_utterances = ['王秀红', '万惠', '马树化', '马正飞', '李桂林', '万惠', '万慧', '张凤珠']
    time_delex = '[时间]'
    place_delex = '[地点]'
    number_delex = '[数字]'
    name_delex = '[人名]'#人名
    company_delex = '[组织名]'

    data = json.loads(open(filename, 'r').read())
    for id, dial in data.items():
        local_kb = []
        removed_turns= []
        #for turn in dial: 
        for turn in dial['log']: # dial in unlabel
            flag = 0
            user = turn['user']
            system = turn['system']
            for it in bad_utterances:
                if (it in user) or (it in system):
                    #dial['log'].remove(turn) # dial.remove(turn)
                    #tmp=dial.remove(turn)
                    removed_turns.append(turn)
                    flag=1
                if flag==1:
                    break
                    #continue
            if flag==1:
                continue
            for it in replace_utterances:
                user = user.replace(it, '')
                system = system.replace(it, '')
                if 'api_result' in turn:
                    turn['api_result'] = turn['api_result'].replace(it, '')
            for it in company_utterances:
                user = user.replace(it, company_delex)
                system = system.replace(it, company_delex)
                if 'api_result' in turn:
                    turn['api_result'] = turn['api_result'].replace(it, company_delex)
            for it in bad_utterances:
                user = user.replace(it, '')
                system = system.replace(it, '')
                if 'api_result' in turn:
                    turn['api_result'] = turn['api_result'].replace(it, '')
            for it in time_utterances:
                user = user.replace(it, time_delex)
                system = system.replace(it, time_delex)
                if 'api_result' in turn:
                    turn['api_result'] = turn['api_result'].replace(it, time_delex)
            for it in time_reg:
                user = re.sub(it, time_delex, user)
                system = re.sub(it, time_delex, system)
                if 'api_result' in turn:
                    turn['api_result'] = re.sub(it, time_delex, turn['api_result'])
            for it in number_utterances:
                user = user.replace(it, number_delex)
                system = system.replace(it, number_delex)
                if 'api_result' in turn:
                    turn['api_result'] = turn['api_result'].replace(it, number_delex)
            for it in number_reg:
                user = re.sub(it, number_delex, user)
                system = re.sub(it, number_delex, system)
                if 'api_result' in turn:
                    turn['api_result'] = re.sub(it, number_delex, turn['api_result'])
            for it in place_utterances:
                user = user.replace(it, place_delex)
                system = system.replace(it, place_delex)
                if 'api_result' in turn:
                    turn['api_result'] = turn['api_result'].replace(it, place_delex)
            for it in name_utterances:
                user = user.replace(it, name_delex)
                system = system.replace(it, name_delex)
                if 'api_result' in turn:
                    turn['api_result'] = turn['api_result'].replace(it, name_delex)

            if 'api_result' in turn:
                api_query = clean_intent(turn['api_query'])
                if '查询用户已办理的业务' in api_query or api_query=='查询流量信息' or api_query=='查询本月月租':
                    if turn['api_result']!="":
                        if '，业务名称：' not in turn['api_result']:
                            local_kb.append(turn['api_result'])
                        else:
                            splitted = turn['api_result'].split('，业务名称：')
                            local_kb.append(splitted[0])
                            for piece in splitted[1:]:
                                local_kb.append(('业务名称：'+piece))
            
            turn['user'] = user
            turn['system'] = system
        
        for t in removed_turns: 
            #dial.remove(t)
            dial['log'].remove(t)
        dial['local_kb'] = local_kb

    json.dump(data, open(new_filename, 'w'), indent=2, ensure_ascii=False)
    return

def merge_annotate_result():
    data_files = {
        'train':['data/ver_0327/train_processed_cln.json', 'data/train.json', 'data/train_final.json'],
        'dev':['data/ver_0327/dev_processed_cln.json', 'data/dev.json', 'data/dev_final.json'],
        'test':['data/ver_0327/test_processed_cln.json', 'data/test.json', 'data/test_final.json']
    }

    special_turns = []

    for dataset, files in data_files.items():
        source_file  = json.loads(open(files[0], 'r').read())
        org_file  = json.loads(open(files[1], 'r').read())
        for id, dial_s in source_file.items():
            dial_org = org_file[id]
            for turn_num in range(len(dial_s['log'])):
                turn_s = dial_s['log'][turn_num]
                if len(dial_org)>turn_num:
                    turn_org= dial_org[turn_num]
                    if turn_org['api_query'] != turn_s['api_query'] and turn_org['user']==turn_s['user']:
                        if turn_org['api_query']!= "查询用户已办理的业务":
                            dial_s['log'][turn_num]['api_query'] = turn_org['api_query'] 
                            dial_s['log'][turn_num]['api_result'] = turn_org['api_result'] 
                        else:
                            special_turn = copy.deepcopy(turn_s)
                            special_turn['api_query_modified'] = turn_org['api_query']
                            special_turn['api_result_modified'] = turn_org['api_result']
                            special_turns.append(special_turn)

        json.dump(source_file, open(files[2], 'w'), indent=2, ensure_ascii=False)
    json.dump(special_turns, open('tmp.json', 'w'), indent=2, ensure_ascii=False)
    return

def get_final_datasets():
    source_data = {'train':['data/train_final_processed.json', 'data/train_final_processed.json'], 
    'dev':['data/dev_final_processed.json', 'data/dev_final_processed.json']}
    #'test':['data/test_final.json', 'data/test_final_processed.json']
    # }
    id_error = []
    global_kb = []
    qa = {}
    for file in source_data:
        count = 0
        dials = json.loads(open(source_data[file][0], 'r').read())
        new_dials = {}
        for id, dial in tqdm(dials.items()):
            local_kb = []
            hist = ['']
            for turn_num in range(len(dial['log'])):
                turn = dial['log'][turn_num]
                api_query = clean_intent(turn['api_query'])
                dial['log'][turn_num]['api_query'] = api_query
                api_results = split_api_result(turn['api_result'])

                if '查询用户已办理的业务' in api_query or api_query=='查询其他信息' or api_query=='查询本月月租' or api_query=='办理' or api_query=='取消' or api_query=='验证' or api_query=='查询流量信息':
                    if turn['api_result']!="":
                        for api_result in api_results:
                            if api_result not in local_kb:
                                local_kb.append(api_result)

                elif api_query!='' and api_query!='[QA]':
                    if turn['api_result']!="":
                        for api_result in api_results:
                            if api_result not in global_kb:
                                global_kb.append(api_result)

                _ , hist = get_spoken(hist, [clean_utt(turn['user'].lower())], role='user')
                if turn['api_query'] == '[QA]':
                    tmp_hist = copy.deepcopy(hist[0])
                    system_cleaned = clean_utt(turn['system'].lower())
                    qa[tmp_hist] = system_cleaned
                _ , hist = get_spoken(hist, [clean_utt(turn['system'].lower())], role='system')  
            new_dial = {'log': dial['log']}
            new_dial['local_kb'] = local_kb
            if local_kb != []:
                count += 1
            new_dials[id] = new_dial

        print(f"file:{file},total_session:{len(dials.keys())},local_kb_num: {count}")
        # json.dump(new_dials, open(source_data[file][1], 'w'), indent=2, ensure_ascii=False) 
    print(id_error)  
    json.dump(global_kb, open('data/global_kb.json', 'w'), indent=2, ensure_ascii=False)
    json.dump(qa, open('data/qa.json', 'w'), indent=2, ensure_ascii=False)
    return

def change_unlabel_id():
    dials = json.loads(open('data/ver_0327/unlabel_cln1.json', 'r').read())
    new_dials = {}
    ids = json.loads(open('data/new_vocab.json', 'r').read())
    for id, dial in dials.items():
        new_dials[ids[id]] = dial
    json.dump(new_dials, open('data/ver_0327/unlabel_cln.json', 'w'), indent=2, ensure_ascii=False)
    return

if __name__ == "__main__":
    #process_annotated_data()
    #analysis_data()
    #analysis_result()
    #clean_qa()
    #clean_ann()
    #get_splitted_data()
    #postprocess_data()
    #get_unlabel_data()
    #remove_sensitive_information()
    turn={
      "user": "江苏省镇江市。镇江技术学校",
      "system": "没查到的。这样，我非常希望能够帮助您，您的诉求，这边反馈上级申请核实处理，请问联系您来电本机，方便接听吗？",
      "api_query": "",
      "api_result": ""
    }
    #from modelscope.pipelines import pipeline
    #from modelscope.utils.constant import Tasks
    #ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_raner_named-entity-recognition_chinese-large-generic')
    #clean_sensitive_model(ner_pipeline, turn)
    #clean_sensitive_manual()
    #clean_bad_utt()
    #merge_annotate_result()
    get_final_datasets()
    #change_unlabel_id()

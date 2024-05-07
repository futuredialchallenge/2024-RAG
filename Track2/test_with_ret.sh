#Copyright 2023 Tsinghua University
#Author: Yucheng Cai (cyc22@mails.tsinghua.edu.cn)
# gt_qa = False
# retrieve_qa = True
python main.py -mode test\
    -cfg device=$1\
    gpt_path=$2\
    rag_testing=True\
    eval_batch_size=8
#Copyright 2023 Tsinghua University
#Author: Yucheng Cai (cyc22@mails.tsinghua.edu.cn)
python main.py -mode train\
    -cfg batch_size=6\
    origin_batch_size=6\
    gradient_accumulation_steps=6\
    epoch_num=40 device=$1\
    exp_name=medium2\
    lr=2e-5
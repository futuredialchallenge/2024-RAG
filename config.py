import logging, time, os

class _Config:
    def __init__(self):
        self.seed=6
        self.exp_name='temp'
        self.exp_path=''
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.mode='train'

        #self.gpt_path ='uer/gpt2-chinese-cluecorpussmall'#can not connected#can not connected
        # uer/gpt2-medium-chinese-cluecorpussmall
        self.gpt = True
        self.gpt_path ='gpt2-medium'
        self.posterior_path ='gpt2-chinese'
        self.t5_path = 'experiments/pretrain_t5/best_model'
        self.t5_posterior_path ='experiments/pretrain_t5/best_model' # 'uer/t5-base-chinese-cluecorpussmall'
        # 'uer/t5-v1_1-base-chinese-cluecorpussmall'
        self.sentence_encoder_path = 'bge-large-zh-v1.5'
        self.api_encoder_path = 'bge-large-zh-v1.5'
        #self.api_encoder_path = 'bge-large-zh-v1.5'
        self.api_save_dir='experiments_retrieve/best_api_model2'
        self.apiret_save_dir='experiments_retrieve/best_apiret_model1'
        self.num_apis = 6
        # bge-large-zh-v1.5
        self.data_path='data/train_final_processed.json'
        self.data_dir='data/'

        self.device=[0]
        self.batch_size=8
        self.origin_batch_size=8
        self.gradient_accumulation_steps=4
        self.epoch_num=40
        self.eval_batch_size=32
        self.lr = 2e-5
        self.warmup_ratio=0.2
        self.pad_id=0
        self.only_target_loss=True
        self.save_type='max_score'
        #self.save_type='min_loss'
        self.debugging = False
        self.train_qa_generation = False

        # config for retrieval
        self.max_length = 512

        # config for jsa
        # unsupervise data proportion in jsa
        self.jsa_unsup = 1
        self.jsa_ebm = False
        self.jsa_ebm_joint = False

        # config for generation
        self.temperature = 1.0
        self.threshold = 0.1
        self.dropout = 0.05
        self.weight_decay = 0.0

        # config for evaluation
        self.gt_db = False
        self.gt_api = True
        self.retrieve_qa = False
        self.retrieve_hist = 3
        self.only_response = True
        self.rag_training = True
        self.rag_testing = False # otherwise use ground truth knowledge
        self.agent_testing = False
        self.response_with_trained_retriever = True
        self.no_retrieval = False
        self.dual_encoder = True
        self.retrieve_with_qa = True

    def _init_logging_handler(self):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists('./log'):
            os.mkdir('./log')
        if 'train' in self.mode:
            file_handler = logging.FileHandler('./log/log_{}_{}_sd{}.txt'.format(self.mode, self.exp_name, self.seed))
            #print(file_handler)
        else:
            file_handler = logging.FileHandler(os.path.join(self.gpt_path, 'eval_log.txt'))
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()
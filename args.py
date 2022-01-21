class Args:
    def __init__(self, task, test_src, min_length, max_length, test_from):
      self.task=task
      self.encoder='bert'
      self.mode='test'
      self.bert_data_path='../bert_data_new/cnndm'
      self.model_path='../models/'
      self.result_path='../results/cnndm'
      self.temp_dir='../temp'
      self.text_src=test_src
      self.text_tgt=''
      self.test_multitask=False
      self.batch_size=140
      self.test_batch_size=200

      self.max_pos=512
      self.use_interval=True
      self.large=False
      self.load_from_extractive=''

      self.sep_optim=False
      self.lr_bert=2e-3
      self.lr_dec=2e-3
      self.use_bert_emb=False

      self.share_emb=False
      self.finetune_bert=True
      self.dec_dropout=0.2
      self.dec_layers=6
      self.dec_hidden_size=768
      self.dec_heads=8
      self.dec_ff_size=2048
      self.enc_hidden_size=512
      self.enc_ff_size=512
      self.enc_dropout=0.2
      self.enc_layers=6

      # params for EXT
      self.ext_dropout=0.2
      self.ext_layers=2
      self.ext_hidden_size=768
      self.ext_heads=8
      self.ext_ff_size=2048

      self.label_smoothing=0.1
      self.generator_shard_size=32
      self.alpha=0.6
      self.beam_size=5
      self.min_length=min_length
      self.max_length=max_length
      self.max_tgt_len=140



      self.param_init=0
      self.param_init_glorot=True
      self.optim='adam'
      self.lr=1
      self.beta1= 0.9
      self.beta2=0.999
      self.warmup_steps=8000
      self.warmup_steps_bert=8000
      self.warmup_steps_dec=8000
      self.max_grad_norm=0

      self.save_checkpoint_steps=5
      self.accum_count=1
      self.report_every=1
      self.train_steps=1000
      self.recall_eval=False


      self.visible_gpus='-1'
      self.gpu_ranks='0'
      self.log_file='../logs/cnndm.log'
      self.seed=666

      self.test_all=False
      self.test_from=test_from
      self.test_start_from=-1

      self.train_from=''
      self.report_rouge=True
      self.block_trigram=True

      self.gpu_ranks = [int(i) for i in range(len('-1'.split(',')))]
      self.world_size = len([int(i) for i in range(len('-1'.split(',')))])



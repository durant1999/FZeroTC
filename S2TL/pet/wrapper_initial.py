# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains code for wrapping a transformer language model and
provides convenience methods for training and inference.
"""
import json
from pyexpat import model
import jsonpickle
import os
from typing import List, Dict, Optional
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, PreTrainedTokenizer, BertForMaskedLM, \
    RobertaForMaskedLM, XLMRobertaForMaskedLM, XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer, \
    XLNetLMHeadModel, BertConfig, BertForSequenceClassification, BertTokenizer, RobertaConfig, \
    RobertaForSequenceClassification, RobertaTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification, \
    XLMRobertaTokenizer, AlbertForSequenceClassification, AlbertForMaskedLM, AlbertTokenizer, AlbertConfig, \
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, RobertaForTokenClassification
from transformers import __version__ as transformers_version

import log
from pet import preprocessor
from pet.tasks import TASK_HELPERS
from pet.utils import InputFeatures, DictDataset, distillation_loss,entropy

logger = log.get_logger('root')
torch.set_printoptions(threshold=np.inf)



CONFIG_NAME = 'wrapper_config.json'
SEQUENCE_CLASSIFIER_WRAPPER = "sequence_classifier"
MLM_WRAPPER = "mlm"
PLM_WRAPPER = "plm"

WRAPPER_TYPES = [SEQUENCE_CLASSIFIER_WRAPPER, MLM_WRAPPER, PLM_WRAPPER]

PREPROCESSORS = {
    SEQUENCE_CLASSIFIER_WRAPPER: preprocessor.SequenceClassifierPreprocessor,
    MLM_WRAPPER: preprocessor.MLMPreprocessor,
    PLM_WRAPPER: preprocessor.PLMPreprocessor,
}

MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: BertForSequenceClassification,
        MLM_WRAPPER: BertForMaskedLM
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: RobertaForSequenceClassification,
        # MLM_WRAPPER: RobertaForMaskedLM
        MLM_WRAPPER: RobertaForTokenClassification
    },
    'xlm-roberta': {
        'config': XLMRobertaConfig,
        'tokenizer': XLMRobertaTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: XLMRobertaForSequenceClassification,
        MLM_WRAPPER: XLMRobertaForMaskedLM
    },
    'xlnet': {
        'config': XLNetConfig,
        'tokenizer': XLNetTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: XLNetForSequenceClassification,
        PLM_WRAPPER: XLNetLMHeadModel
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: AlbertForSequenceClassification,
        MLM_WRAPPER: AlbertForMaskedLM
    },
    'gpt2': {
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        MLM_WRAPPER: GPT2LMHeadModel
    },
}

EVALUATION_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_eval_step,
    PLM_WRAPPER: lambda wrapper: wrapper.plm_eval_step,
    SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_eval_step,
}

TRAIN_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_train_step,
    PLM_WRAPPER: lambda wrapper: wrapper.plm_train_step,
    SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_train_step,
}


class WrapperConfig(object):
    """A configuration for a :class:`TransformerModelWrapper`."""

    def __init__(self, model_type: str, model_name_or_path: str, wrapper_type: str, task_name: str, max_seq_length: int,
                 label_list: List[str], pattern_id: int = 0, verbalizer_file: str = None, cache_dir: str = None):
        """
        Create a new config.

        :param model_type: the model type (e.g., 'bert', 'roberta', 'albert')
        :param model_name_or_path: the model name (e.g., 'roberta-large') or path to a pretrained model
        :param wrapper_type: the wrapper type (one of 'mlm', 'plm' and 'sequence_classifier')
        :param task_name: the task to solve
        :param max_seq_length: the maximum number of tokens in a sequence
        :param label_list: the list of labels for the task
        :param pattern_id: the id of the pattern to use
        :param verbalizer_file: optional path to a verbalizer file
        :param cache_dir: optional path to a cache dir
        """
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.wrapper_type = wrapper_type
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.pattern_id = pattern_id
        self.verbalizer_file = verbalizer_file
        self.cache_dir = cache_dir
        #loss
        # self.loss_s=[]
        # self.loss_s_with_uncertainty=[]
        # self.loss_p=[]
        # self.loss_kl=[]
        # self.loss_total=[]


class MyModel(nn.Module):
    def __init__(self,hidden_size,num_labels,model_name_or_path,model_config,cache_dir):
        super().__init__()
        self.roberta_for_masked_lm = RobertaForMaskedLM.from_pretrained(model_name_or_path,config=model_config,cache_dir=cache_dir)
        self.dense=self.roberta_for_masked_lm.lm_head.dense
        self.layer_norm=self.roberta_for_masked_lm.lm_head.layer_norm
        self.gelu=nn.functional.gelu
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):
       outputs=self.roberta_for_masked_lm(input_ids,output_hidden_states=True)
       x=self.dense(outputs[1][-1])
       x=self.gelu(x)
       x=self.layer_norm(x)
       logits=self.classifier(x)
       
       return (logits,outputs[1],outputs[0])
       
            

class TransformerModelWrapper:
    """A wrapper around a Transformer-based language model."""

    def __init__(self, config: WrapperConfig):
        """Create a new wrapper from the given config."""
        # loss 
        self.loss_dict={"loss_s": [], "loss_s_with_uncertainty": [], "loss_p": [], "loss_kl": [], "loss_total": [], "loss_on_labeled": []}

        self.config = config
        config_class = MODEL_CLASSES[self.config.model_type]['config']
        tokenizer_class = MODEL_CLASSES[self.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[self.config.model_type][self.config.wrapper_type]
        
        self.limited_clusters=4
        
        
        model_config = config_class.from_pretrained(
            config.model_name_or_path, num_labels=self.limited_clusters, finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None, use_cache=False)

        self.tokenizer = tokenizer_class.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None)  # type: PreTrainedTokenizer

        if self.config.model_type == 'gpt2':
            self.tokenizer.pad_token, self.tokenizer.mask_token = self.tokenizer.eos_token, self.tokenizer.eos_token


        self.masked_lm_model= RobertaForMaskedLM.from_pretrained(config.model_name_or_path,config=model_config,cache_dir=config.cache_dir if config.cache_dir else None)

        self.model = model_class.from_pretrained(config.model_name_or_path, config=model_config,
                                                 cache_dir=config.cache_dir if config.cache_dir else None)
        
        # Multi GPU Training
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.preprocessor = PREPROCESSORS[self.config.wrapper_type](self, self.config.task_name, self.config.pattern_id,
                                                                    self.config.verbalizer_file)
        self.task_helper = TASK_HELPERS[self.config.task_name](
            self) if self.config.task_name in TASK_HELPERS else None
        
        # representation pool
        self.masked_embedding_pool = None
        self.masked_embedding_pool_lm = None
        self.pred_soft_label_pool = None 
        self.true_label_pool = [] 
        self.vocab_soft_label_pool=None
        self.top30_for_each_sample=[]
        
        # match acc:
        self.match_pair_counts=0
        # match dict:
        self.match_dict= dict()
        self.samples_stat_dict=dict()
        self.samples_nearest_neighbors=list()
        self.limited_cluster_dict={i:[] for i in range(self.limited_clusters)}
        self.fnn_cluster_list=[]
        self.vocab_embedding=self.masked_lm_model.lm_head
        W2=list(self.model.classifier.named_parameters())[0][1]
        self.uncertainty=1
        
        

    @classmethod
    def from_pretrained(cls, path: str) -> 'TransformerModelWrapper':
        """Load a pretrained wrapper from a given path."""
        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)
        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[wrapper.config.model_type][wrapper.config.wrapper_type]
        wrapper.model = model_class.from_pretrained(path)
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)
        wrapper.preprocessor = PREPROCESSORS[wrapper.config.wrapper_type](
            wrapper, wrapper.config.task_name, wrapper.config.pattern_id, wrapper.config.verbalizer_file)
        wrapper.task_helper = TASK_HELPERS[wrapper.config.task_name](wrapper) \
            if wrapper.config.task_name in TASK_HELPERS else None
        return wrapper

    def save(self, path: str) -> None:
        """Save a pretrained wrapper."""
        model_to_save = self.model.module if hasattr(
            self.model, 'module') else self.model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), 'w') as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), 'r') as f:
            return jsonpickle.decode(f.read())

    def train(self, task_train_data: List[InputExample], device, per_gpu_train_batch_size: int = 8, n_gpu: int = 1,
              num_train_epochs: int = 3, gradient_accumulation_steps: int = 1, weight_decay: float = 0.0,
              learning_rate: float = 5e-5, adam_epsilon: float = 1e-8, warmup_steps=0, max_grad_norm: float = 1,
              logging_steps: int = 50, per_gpu_unlabeled_batch_size: int = 8, unlabeled_data: List[InputExample] = None,
              lm_training: bool = False, use_logits: bool = False, alpha: float = 0.8, temperature: float = 1,
              max_steps=-1, **_):
        """
        Train the underlying language model.

        :param task_train_data: the training examples to use
        :param device: the training device (cpu/gpu)
        :param per_gpu_train_batch_size: the number of training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train
        :param gradient_accumulation_steps: the number of gradient accumulation steps before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the learning rate to use
        :param adam_epsilon: epsilon parameter for the Adam optimizer
        :param warmup_steps: the number of warmup steps
        :param max_grad_norm: the maximum norm for the gradient
        :param logging_steps: the number of steps after which logging information is printed
        :param per_gpu_unlabeled_batch_size: the number of unlabeled examples per batch and gpu
        :param unlabeled_data: the unlabeled examples to use
        :param lm_training: whether to perform auxiliary language modeling (only for MLMs)
        :param use_logits: whether to use the example's logits instead of their labels to compute the loss
        :param alpha: the alpha parameter for auxiliary language modeling #################### important
        :param temperature: the temperature for knowledge distillation
        :param max_steps: the maximum number of training steps, overrides ``num_train_epochs`` ###########
        :return: a tuple consisting of the total number of steps and the average training loss
        """

        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        train_dataset = self._generate_dataset(task_train_data)
        train_sampler=None
        if len(task_train_data)!=0  :
            train_sampler = SequentialSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        unlabeled_dataloader, unlabeled_iter = None, None

        if lm_training or use_logits: 
            # we need unlabeled data both for auxiliary language modeling and for knowledge distillation
            assert unlabeled_data is not None
            unlabeled_batch_size = per_gpu_unlabeled_batch_size * max(1, n_gpu)
            unlabeled_dataset = self._generate_dataset(
                unlabeled_data, labelled=True)
            unlabeled_sampler=SequentialSampler(unlabeled_dataset)
            unlabeled_dataloader = DataLoader(unlabeled_dataset, sampler=unlabeled_sampler,
                                              batch_size=unlabeled_batch_size)
            unlabeled_iter = unlabeled_dataloader.__iter__()

        if use_logits:
            train_dataloader = unlabeled_dataloader

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (
                max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1
        else:
            t_total = len(
                train_dataloader) // gradient_accumulation_steps * num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        step = 0
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()

        train_iterator = trange(int(num_train_epochs), desc="Epoch")

        self_defined_epoch=0
        
        for _ in train_iterator: #epochs
            unlabeled_pool_iterator = tqdm(unlabeled_dataloader, desc="update pool")
            self.masked_embedding_pool = None
            self.pred_soft_label_pool = None 
            self.true_label_pool = None
            self.top30_for_each_sample=[] 
            self.match_pair_counts=0
            self.match_dict= dict()
            self.samples_stat_dict=dict()
            self.samples_nearest_neighbors=list()
            self.limited_cluster_dict={i:[] for i in range(self.limited_clusters)}
            self.fnn_cluster_list=[]
            print("pool update start")
            for _, batch in enumerate(unlabeled_pool_iterator):
                self.model.eval()
                
                unlabeled_batch = {k: t.to(device) for k, t in batch.items()}
                unlabeled_inputs=self.generate_default_inputs(unlabeled_batch)
                unlabeled_outputs=self.model(**unlabeled_inputs,output_hidden_states=True)
                
                self.masked_lm_model.to(torch.device("cuda"))
                unlabeled_outputs_lm=self.masked_lm_model(**unlabeled_inputs,output_hidden_states=True)
                unlabeled_fea_lm=unlabeled_outputs_lm[1][-1][unlabeled_batch['mlm_labels']>=0].detach().cpu()
                
                unlabeled_pred_logits=unlabeled_outputs[0][unlabeled_batch['mlm_labels']>=0].detach().cpu() # logits (batchsize,clusters_size)
                unlabeled_pred_logits=F.softmax(unlabeled_pred_logits,dim=1)
                unlabeled_fea=unlabeled_outputs[1][-1][unlabeled_batch['mlm_labels']>=0].detach().cpu() # embedding  （batchsize,768）

                if self.masked_embedding_pool==None:
                    self.masked_embedding_pool=unlabeled_fea
                    self.masked_embedding_pool_lm=unlabeled_fea_lm
                    self.pred_soft_label_pool=unlabeled_pred_logits
                    self.true_label_pool=unlabeled_batch['labels'].detach().cpu()
                else:
                    self.masked_embedding_pool=torch.cat([self.masked_embedding_pool,unlabeled_fea],0)
                    self.masked_embedding_pool_lm=torch.cat([self.masked_embedding_pool_lm,unlabeled_fea_lm],0)
                    
                    self.pred_soft_label_pool=torch.cat([self.pred_soft_label_pool,unlabeled_pred_logits],0)
                    self.true_label_pool=torch.cat([self.true_label_pool,unlabeled_batch['labels'].detach().cpu()],0)
            self.masked_embedding_pool= self.masked_embedding_pool / torch.norm(self.masked_embedding_pool, 2, 1, keepdim=True)
            self.masked_embedding_pool_lm= self.masked_embedding_pool_lm / torch.norm(self.masked_embedding_pool_lm, 2, 1, keepdim=True)
            self.uncertainty=self.cal_uncertainty(self.pred_soft_label_pool)
            W2=list(self.model.classifier.named_parameters())[0][1]
            for i,masked_embedding in enumerate(self.masked_embedding_pool):
                K=30
                self.vocab_embedding=self.vocab_embedding.to(torch.device("cpu"))
                vocab_logits= self.vocab_embedding(masked_embedding.reshape(-1,768))[0]
                top_K_prob_word_indices=torch.topk(vocab_logits,K).indices
                top_K_prob_words=[self.tokenizer.decode([x]) for x in top_K_prob_word_indices.tolist()]
                rank_str=[]
                for rank,prob_word in enumerate(top_K_prob_words):
                    rank_str.append((rank+1,prob_word))
                print(rank_str)
                self.top30_for_each_sample.append(rank_str)
            
            if self_defined_epoch!=0:    
                epoch_iterator = tqdm(train_dataloader, desc="Iteration")
                for _, batch in enumerate(epoch_iterator):
                    self.model.train()
                    unlabeled_batch = None
                    lm_unlabeled_batch = None

                    batch = {k: t.to(device) for k, t in batch.items()}
                    if lm_training:
                        while unlabeled_batch is None:
                            try:
                                unlabeled_batch = unlabeled_iter.__next__()
                            except StopIteration:
                                logger.info("Resetting unlabeled dataset")
                                unlabeled_iter = unlabeled_dataloader.__iter__()

                        lm_unlabeled_batch=None
                        # pairwise data
                        unlabeled_batch = {k: t.to(device)
                                        for k, t in unlabeled_batch.items()}

                    train_step_inputs = {
                        'unlabeled_batch': unlabeled_batch, 'lm_training': lm_training, 'alpha': alpha,
                        "lm_unlabeled_batch":lm_unlabeled_batch, 'use_logits': use_logits, 'temperature': temperature
                    }
                    loss = self.task_helper.train_step(
                        batch, **train_step_inputs) if self.task_helper else None
                    if loss is None:
                        loss = TRAIN_STEP_FUNCTIONS[self.config.wrapper_type](
                            self)(batch, **train_step_inputs)

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps

                    loss.backward() 

                    tr_loss += loss.item()
                    if (step + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        self.model.zero_grad()
                        global_step += 1

                        if logging_steps > 0 and global_step % logging_steps == 0:
                            logs = {}
                            loss_scalar = (tr_loss - logging_loss) / logging_steps
                            learning_rate_scalar = scheduler.get_lr()[0]
                            logs['learning_rate'] = learning_rate_scalar
                            logs['loss'] = loss_scalar
                            logging_loss = tr_loss

                            print(json.dumps({**logs, **{'step': global_step}}))

                    if 0 < max_steps < global_step:
                        epoch_iterator.close()
                        break
                    step += 1
                if 0 < max_steps < global_step:
                    train_iterator.close()
                    break
            
            
            unlabeled_pool_iterator = tqdm(unlabeled_dataloader, desc="cal match pool")
            for _, batch in enumerate(unlabeled_pool_iterator):
                self.model.eval()
                unlabeled_batch = {k: t.to(device) for k, t in batch.items()}
                self.cal_match_stats(unlabeled_batch)
            


            print("current epoch pair match counts: {}/{} . ".format(self.match_pair_counts,self.masked_embedding_pool.shape[0]))
            for key,val in self.match_dict.items():
                print("current epoch match on class {} is {}/{}: ".format(key,val,self.samples_stat_dict[key]))  

            self.cluster(self_defined_epoch)
            self_defined_epoch+=1
            
            print("epoch {} after fnn cluster stats: ".format(self_defined_epoch-1))
            for i,val in enumerate(self.fnn_cluster_list):
                self.limited_cluster_dict[val].append(i)
            for i in range(self.limited_clusters):
                print("class {} contains samples {}".format(i,self.limited_cluster_dict[i]))
                tmp_true_labels=self.true_label_pool[self.limited_cluster_dict[i]]
                if len(self.limited_cluster_dict[i])!=0:
                    tmp_mode= torch.mode(tmp_true_labels).values
                    tmp_mode_count= tmp_true_labels.eq(tmp_mode).sum().item()
                else :
                    tmp_mode_count=0
                print("         but their true labels is {}, the purity is {}/{}".format(tmp_true_labels,tmp_mode_count,len(self.limited_cluster_dict[i])))
                topic_words_dict={}
                for idx in self.limited_cluster_dict[i]:
                    top30_list=self.top30_for_each_sample[idx]
                    for item in top30_list:
                        if item[1] not in topic_words_dict:
                            topic_words_dict[item[1]]=0
                        topic_words_dict[item[1]]+=1
                max_words=min(len(topic_words_dict),20)
                
                print("class {} topic words may contain {}".format(i,sorted(topic_words_dict.items(),key=lambda x:x[1],reverse=True)[:max_words]))
                print() 
            
            for i,embedding in enumerate(self.masked_embedding_pool):
                print(embedding.equal(self.masked_embedding_pool_lm[i]),torch.dot(embedding,self.masked_embedding_pool_lm[i]))
                
            
        return global_step, (tr_loss / global_step if global_step > 0 else -1)

    def cluster(self,epoch):
        print("====================cluster result===================")
        print("neighbors: ",self.samples_nearest_neighbors)
        parents=[i for i in range(len(self.samples_nearest_neighbors))]
        for i in range(len(parents)):
            i_root=parents[i]
            neighbor_root=parents[self.samples_nearest_neighbors[i]]
            if i_root == neighbor_root:
                continue
            elif i_root < neighbor_root:
                parents[neighbor_root]=i_root
            else:
                parents[i_root]=neighbor_root
                                
        cluster_dict=dict()
        for i in range(len(parents)):
            tmp=parents[i]
            while tmp!=parents[tmp]:
                tmp=parents[tmp]
                
            if tmp not in cluster_dict:
                cluster_dict[tmp]=[]
            cluster_dict[tmp].append(i)
            
        counts=0
        purity_counts=0
        for key,set_val in cluster_dict.items():
            set_val.sort()
            tmp_true_labels=self.true_label_pool[set_val]
            tmp_mode= torch.mode(tmp_true_labels).values
            tmp_mode_count= tmp_true_labels.eq(tmp_mode).sum().item()
            print("cluster {} contain smaples     : {}".format(key,set_val))
            print("cluster {} samples true labels : {} , purity is {}/{}={}".format(key,tmp_true_labels,tmp_mode_count,len(set_val),tmp_mode_count/len(set_val)))
            
            counts+=len(set_val)
            purity_counts+=tmp_mode_count
        print("epoch {}    clusters: {}, total samples: {}, total purity: {}/{}={}".format(epoch,len(cluster_dict),counts,purity_counts,counts,purity_counts/counts))
        
                    


    def eval(self, eval_data: List[InputExample], device, per_gpu_eval_batch_size: int = 8, n_gpu: int = 1,
             priming: bool = False, decoding_strategy: str = 'default') -> Dict:
        """
        Evaluate the underlying language model.

        :param eval_data: the evaluation examples to use
        :param device: the evaluation device (cpu/gpu)
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param priming: whether to use priming
        :param decoding_strategy: the decoding strategy for PET with multiple masks ('default', 'ltr' or 'parallel')
        :return: a dictionary of numpy arrays containing the indices, logits, labels, and (optional) question_ids for
                 each evaluation example.
        """

        eval_dataset = self._generate_dataset(eval_data, priming=priming)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        preds = None
        all_indices, out_label_ids, question_ids = None, None, None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()

            batch = {k: t.to(device) for k, t in batch.items()}
            labels = batch['labels']
            indices = batch['idx']
            with torch.no_grad():

                # some tasks require special evaluation
                logits = self.task_helper.eval_step(batch,
                                                    decoding_strategy=decoding_strategy) if self.task_helper else None

                if logits is None:
                    logits = EVALUATION_STEP_FUNCTIONS[self.config.wrapper_type](
                        self)(batch)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
                if 'question_idx' in batch:
                    question_ids = batch['question_idx'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0)
                all_indices = np.append(
                    all_indices, indices.detach().cpu().numpy(), axis=0)
                if 'question_idx' in batch:
                    question_ids = np.append(
                        question_ids, batch['question_idx'].detach().cpu().numpy(), axis=0)

        return {
            'indices': all_indices,
            'logits': preds,
            'labels': out_label_ids,
            'question_ids': question_ids
        }

    def _generate_dataset(self, data: List[InputExample], labelled: bool = True, priming: bool = False):
        features = self._convert_examples_to_features(
            data, labelled=labelled, priming=priming)
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long),
            'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
            'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
            'idx': torch.tensor([f.idx for f in features], dtype=torch.long)
        }
        if self.config.wrapper_type == PLM_WRAPPER:
            feature_dict['perm_mask'] = torch.tensor(
                [f.perm_mask for f in features], dtype=torch.float)
            feature_dict['target_mapping'] = torch.tensor(
                [f.target_mapping for f in features], dtype=torch.float)

        if self.task_helper:
            self.task_helper.add_features_to_dict(features, feature_dict)

        return DictDataset(**feature_dict)

    def _convert_examples_to_features(self, examples: List[InputExample], labelled: bool = True,
                                      priming: bool = False) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example {}".format(ex_index))
            input_features = self.preprocessor.get_input_features(
                example, labelled=labelled, priming=priming)
            if self.task_helper:
                self.task_helper.add_special_input_features(
                    example, input_features)
            features.append(input_features)
            if ex_index < 5:
                logger.info(f'--- Example {ex_index} ---')
                logger.info(input_features.pretty_print(self.tokenizer))
        return features

    def _mask_tokens(self, input_ids):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = input_ids.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                               labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(
            special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # if a version of transformers < 2.4.0 is used, -1 is the expected value for indices to ignore
        if [int(v) for v in transformers_version.split('.')][:3] >= [2, 4, 0]:
            ignore_value = -100
        else:
            ignore_value = -1

        # We only compute loss on masked tokens
        labels[~masked_indices] = ignore_value

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels

    def generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate the default inputs required by almost every language model."""
        inputs = {'input_ids': batch['input_ids'],
                  'attention_mask': batch['attention_mask']}
        if self.config.model_type in ['bert', 'xlnet']:
            inputs['token_type_ids'] = batch['token_type_ids']
        return inputs

    def cal_uncertainty(self,all_data_prob:torch.Tensor):
        conf, _ =all_data_prob.max(1)
        # conf=conf.detach().cpu().numpy()
        conf=conf.numpy()
        mean_uncert=1-np.mean(conf)
        # print('conf :',conf)
        print('mean_uncert :',mean_uncert)
        
        return mean_uncert
    
    def cal_loss_with_uncertainty(self, x: torch.Tensor, target: torch.tensor, uncertainty: float):
        scale_index=10
        uncertainty=min(uncertainty,0.5)
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m=x + uncertainty * scale_index 
        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target)



    def cal_match_stats(self, unlabeled_batch: Dict[str, torch.Tensor]):
        unlabeled_inputs=self.generate_default_inputs(unlabeled_batch)                
        unlabeled_outputs=self.model(**unlabeled_inputs,output_hidden_states=True)   
        unlabeled_pred_logits=unlabeled_outputs[0][unlabeled_batch['mlm_labels']>=0] #prob  logits
        unlabeled_fea=unlabeled_outputs[1][-1][unlabeled_batch['mlm_labels']>=0] # embedding

        max_idxs=unlabeled_pred_logits.argmax(dim=1).cpu().numpy().tolist()
        self.fnn_cluster_list.extend(max_idxs)
        
        unlabeled_batch_fea=unlabeled_fea.detach().cpu()
        unlabeled_batch_fea_norm=unlabeled_batch_fea / torch.norm(unlabeled_batch_fea, 2, 1, keepdim=True)
        
        unlabel_cosine_dist=torch.mm(unlabeled_batch_fea_norm,self.masked_embedding_pool.t())
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
        pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
        pos_pairs=pos_idx
        self.samples_nearest_neighbors.extend(pos_pairs)
      
        match_counts=0
        for i in range(unlabeled_batch['labels'].shape[0]):
            tmp=unlabeled_batch['labels'][i].item()
            if tmp not in self.samples_stat_dict:
                self.samples_stat_dict[tmp]=0
            self.samples_stat_dict[tmp]+=1
            # match
            if tmp==self.true_label_pool[pos_pairs][i].item():
                if tmp not in self.match_dict:
                    self.match_dict[tmp]=0     
                self.match_dict[tmp]+=1
                match_counts+=1
        print("match counts: ",match_counts)       
        self.match_pair_counts+=match_counts




    def mlm_train_step(self, labeled_batch: Dict[str, torch.Tensor],
                       unlabeled_batch: Optional[Dict[str, torch.Tensor]] = None, lm_training: bool = False,
                       lm_unlabeled_batch: Optional[Dict[str, torch.Tensor]] = None, alpha: float = 0, **_) -> torch.Tensor:
        """Perform a MLM training step."""
        inputs = self.generate_default_inputs(labeled_batch)
        mlm_labels, labels = labeled_batch['mlm_labels'], labeled_batch['labels']
        
        outputs = self.model(**inputs,output_hidden_states=True)  #outputs[]
        labeled_pred_logits=outputs[0][mlm_labels >= 0]  #logits
        
        loss_s = nn.CrossEntropyLoss()(labeled_pred_logits.view(-1, self.limited_clusters),
                                     labels.view(-1))
        
        unlabeled_inputs=self.generate_default_inputs(unlabeled_batch)
        unlabeled_outputs=self.model(**unlabeled_inputs,output_hidden_states=True)   
        unlabeled_pred_logits=unlabeled_outputs[0][unlabeled_batch['mlm_labels']>=0] #prob  logits
        unlabeled_fea=unlabeled_outputs[1][-1][unlabeled_batch['mlm_labels']>=0] # embedding
        unlabeled_pred_logits=F.softmax(unlabeled_pred_logits,dim=1)
        unlabeled_batch_fea=unlabeled_fea.detach().cpu()
        unlabeled_batch_fea_norm=unlabeled_batch_fea / torch.norm(unlabeled_batch_fea, 2, 1, keepdim=True)
        unlabel_cosine_dist=torch.mm(unlabeled_batch_fea_norm,self.masked_embedding_pool.t())
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
        pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
        pos_pairs=pos_idx
        
        pos_prob =self.pred_soft_label_pool[pos_pairs].to("cuda")
        cur_batch_size=unlabeled_outputs[0].shape[0]
        pos_sim = torch.bmm(unlabeled_pred_logits.view(cur_batch_size, 1, -1),
                            pos_prob.view(cur_batch_size, -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        bce = nn.BCELoss()
        # pairwise loss
        loss_p = bce(pos_sim, ones) 
        loss_kl = -entropy(torch.mean(unlabeled_pred_logits, 0)) 

        loss = loss_s + loss_p + 5*loss_kl
        self.loss_dict['loss_s'].append(loss_s.item())
        self.loss_dict['loss_p'].append(loss_p.item())
        self.loss_dict['loss_kl'].append(loss_kl.item())
        self.loss_dict['loss_total'].append(loss.item())

        return loss

    def plm_train_step(self, labeled_batch: Dict[str, torch.Tensor], lm_training: bool = False, **_):
        """Perform a PLM training step."""

        inputs = self.generate_default_inputs(labeled_batch)
        inputs['perm_mask'], inputs['target_mapping'] = labeled_batch['perm_mask'], labeled_batch['target_mapping']
        labels = labeled_batch['labels']
        outputs = self.model(**inputs)
        prediction_scores = self.preprocessor.pvp.convert_plm_logits_to_cls_logits(
            outputs[0])
        loss = nn.CrossEntropyLoss()(
            prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))

        if lm_training:
            raise NotImplementedError(
                "Language model training is currently not implemented for PLMs")

        return loss

    def sequence_classifier_train_step(self, batch: Dict[str, torch.Tensor], use_logits: bool = False,
                                       temperature: float = 1, **_) -> torch.Tensor:
        """Perform a sequence classifier training step."""

        inputs = self.generate_default_inputs(batch)
        if not use_logits:
            inputs['labels'] = batch['labels']

        outputs = self.model(**inputs)

        if use_logits:
            logits_predicted, logits_target = outputs[0], batch['logits']
            return distillation_loss(logits_predicted, logits_target, temperature)
        else:
            return outputs[0]

    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM evaluation step."""
        inputs = self.generate_default_inputs(batch)
        # print("batch data is:",batch)
        outputs = self.model(**inputs) 
        return self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])

    def plm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a PLM evaluation step."""
        inputs = self.generate_default_inputs(batch)
        inputs['perm_mask'], inputs['target_mapping'] = batch['perm_mask'], batch['target_mapping']
        outputs = self.model(**inputs)
        return self.preprocessor.pvp.convert_plm_logits_to_cls_logits(outputs[0])

    def sequence_classifier_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a sequence classifier evaluation step."""
        inputs = self.generate_default_inputs(batch)
        return self.model(**inputs)[0]

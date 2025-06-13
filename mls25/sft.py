import os, sys, logging
import time
import json
from glob import glob
import pandas as pd
from collections import defaultdict
from copy import deepcopy
import torch
import numpy as np
from tqdm import tqdm

import util
args = util.parser.parse_args()
if args.use_unsloth:
    try:
        from unsloth import FastLanguageModel
    except Exception as e:
        print(e)


from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Union
import librosa

#import trl
#from trl import DataCollatorForCompletionOnlyLM

from sklearn.model_selection import train_test_split, GroupKFold, KFold, StratifiedKFold
from types import MethodType
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import RandomSampler, default_collate
from transformers import Trainer as HFTrainer, TrainingArguments as HFTrainingArguments
from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_utils import has_length
from transformers import AutoProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import ROPE_INIT_FUNCTIONS
from transformers import DebertaV2ForSequenceClassification
from transformers import LlamaForSequenceClassification, Qwen2ForSequenceClassification, Gemma2ForSequenceClassification, Qwen2Config
from transformers.models.t5 import T5ForSequenceClassification

from pt_util import set_seed
import pt_util as pu

from dataset import DatasetMix, IterMixBase

logger = logging.getLogger(__name__)



if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

trans_ppt = "请将下面的中文电视剧对话翻译成{}：{}"
trans_ppt2 = "请将下面的{}翻译成{}：{}"
qw2audio_asrppt = "<|audio_bos|><|AUDIO|><|audio_eos|>" + "Detect the language and recognize the speech: <|zh|>{}<|endoftext|>"
qw2audio_sttppt = "<|audio_bos|><|AUDIO|><|audio_eos|>" + "Detect the language and translate the speech into {}: <|zh|>{}<|endoftext|>"
lanencodes = {"马来语":"Malay", "英语":"English", "泰语":"Thai"}

class Sampler(torch.utils.data.Sampler):
    def __init__(self, cfg, data_type, ds):
        self.cfg = cfg
        self.data_type = data_type
        self.ds = ds
        self.inds = np.arange(len(ds))
        assert len(self.inds) == len(self.ds.data)
        title_num = defaultdict(int)
        for rec in self.ds.data:
            title_num[rec.title] += 1
        title_weight = {k:1/v for k, v in title_num.items()}
        self.weights = [title_weight[rec.title] for rec in self.ds.data]
        self.weights = np.array(self.weights)/np.sum(self.weights)

        assert abs(1-sum(self.weights))<1e-10

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        for ind in self.gen_inds():
            yield ind

    def gen_inds(self):
        if self.data_type=='train':
            inds = np.random.choice(self.inds, self.__len__(), p=self.weights)
        else:
            raise NotImplementedError(self.data_type)
        return inds



def gen_ds(args, data_type, data, **kwargs):
    drop_last, shuffle, num_workers, sampler, batch_size, collate_func = False, False, args.n_dl_worker, None, args.batch_size, None
    if data_type=='train':
        ds_cls = globals()[args.ds_cls]
        drop_last, shuffle = True, True
    elif data_type=='val':
        ds_cls = globals()[args.val_ds_cls]
        batch_size = args.val_batch_size
    else:
        ds_cls = globals()[args.test_ds_cls]
        batch_size = args.val_batch_size
    ds = ds_cls(args, data_type, data, **kwargs)
    collate_func = ds.collate

    if args.use_sampler and data_type == 'train':
        sampler = Sampler(args, data_type, ds)
        shuffle = False

    return ds


class Dataset(DatasetMix, torch.utils.data.Dataset):
    def __init__(self, cfg, data_type, data, tokenizer=None, model_config=None):
        super().__init__(cfg, data_type, data, tokenizer=tokenizer, model_config=model_config)
        self.ppt = globals()[self.cfg.ppt]
        logger.info('use ppt:%s', self.ppt)

    def get_start_id(self, msg):
        msg = [*msg,
            {"role": "assistant", "content": f"{self.tokenizer.eos_token}" * 8}
        ]
        target = [self.tokenizer.eos_token_id]*8
        input_ids = self.tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=False, enable_thinking=False)
        for i in range(len(input_ids)):
            if np.all(input_ids[i:i+8]==target):
                break
        return i


    def get_input_ids(self, rec, index):
        text = rec['中文']
        language = '中文'
        tgt_text = rec['文本']
        tgt_language = rec['语言']
        if self.data_type=='train' and self.cfg.aug_zh>0 and np.random.rand()<self.cfg.aug_zh:
            text, tgt_text, language, tgt_language = tgt_text, text, tgt_language, language
        if self.cfg.ppt=='trans_ppt2':
             messages = [
                 {"role": "user", "content": self.ppt.format(language, tgt_language, text)}
             ]
        elif self.cfg.ppt=='trans_ppt':
            messages = [
                {"role": "user", "content": self.ppt.format(tgt_language, text)}
            ]
        else:
            raise NotImplementedError(self.cfg.ppt)
        start = self.get_start_id(messages)
        messages.append({"role": "assistant", "content":tgt_text})
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
        )
        labels = np.array(deepcopy(input_ids))
        labels[:start] = -100
        return input_ids, labels


    def _getitem(self, index, rec=None):
        if rec is None:
            rec = self.data[index]
        input_ids, labels = self.get_input_ids(rec, index)
        input_len = len(input_ids)
        item = dict(input_ids=input_ids, labels=labels)
        item['seq_len'] = input_len
        item['input_ids'] = np.array(input_ids)
        item['attention_mask'] = np.ones([len(input_ids)])
        return item

    def collate(self, batch):
        new_batch = dict()
        for k in ['ID']:
            if k in batch[0]:
                new_batch[k] = [item.pop(k) for item in batch]

        input_lens = [item['seq_len'] for item in batch]
        max_len = max(input_lens)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        for i, item in enumerate(batch):
            if self.tokenizer.padding_side == 'left':
                item['input_ids'] = np.pad(item['input_ids'], ((max_len - item['seq_len'], 0)), "constant", constant_values=pad_token_id)
                if 'attention_mask' in item:
                    item['attention_mask'] = np.pad(item['attention_mask'], ((max_len - item['seq_len'], 0)), "constant", constant_values=0)
                if 'labels' in item:
                    item['labels'] = np.pad(item['labels'], ((max_len - item['seq_len'], 0)), "constant", constant_values=-100)
            else:
                item['input_ids'] = np.pad(item['input_ids'], ((0, max_len - item['seq_len'])), "constant", constant_values=pad_token_id)
                if 'attention_mask' in item:
                    item['attention_mask'] = np.pad(item['attention_mask'], ((0, max_len - item['seq_len'])), "constant", constant_values=0)
                if 'labels' in item:
                    item['labels'] = np.pad(item['labels'], ((0, max_len - item['seq_len'])), "constant", constant_values=-100)
        batch = default_collate(batch)
        batch.update(new_batch)
        return batch


class ASRLMDataset(Dataset):
    def __init__(self, cfg, data_type, data, tokenizer=None, model_config=None):
        super().__init__(cfg, data_type, data, tokenizer=tokenizer, model_config=model_config)
        self.processor = AutoProcessor.from_pretrained(self.cfg.backbone, trust_remote_code=True)
        self.sr = self.processor.feature_extractor.sampling_rate
        self.start_token_id = self.processor.tokenizer.convert_tokens_to_ids(["<|zh|>"])[-1]

    def preprocess_data(self, data):
        data = data[~data['编号'].isin([8334, 8335, 48749, 48750])]
        data = data.groupby(["音频路径"]).head(1).reset_index(drop=True)
        data = data.to_records(index=False)
        logger.info('num of data %s is %s', self.data_type, len(data))
        return data

    def get_input(self, rec, index):
        text = self.ppt.format(rec['中文'])
        audio, sr = librosa.load(rec['音频路径'], sr=self.processor.feature_extractor.sampling_rate)
        return audio, text


    def _getitem(self, index, rec=None):
        if rec is None:
            rec = self.data[index]
        audio, text = self.get_input(rec, index)
        item = dict(audio=audio, text=text)
        return item

    def collate(self, batch):
        new_batch = dict()
        for k in ['ID', 'audio', 'text']:
            if k in batch[0]:
                new_batch[k] = [item.pop(k) for item in batch]

        batch = self.processor(text=new_batch.pop('text'), audio=new_batch.pop('audio'), return_tensors='pt', padding=True, audio_kwargs=dict(sampling_rate=self.processor.feature_extractor.sampling_rate))
        labels = deepcopy(batch['input_ids'])
        inds = torch.where(labels==self.start_token_id)[-1]
        for i, ind in enumerate(inds):
            labels[i][:ind+1] = -100
        batch['labels'] = labels
        batch.update(new_batch)
        return batch


class STTDataset(ASRLMDataset):
    def preprocess_data(self, data):
        data = data[~data['编号'].isin([8334, 8335, 48749, 48750])]
        data = data.to_records(index=False)
        logger.info('num of data %s is %s', self.data_type, len(data))
        return data

    def get_input(self, rec, index):
        text = self.ppt.format(lanencodes[rec['语言']], rec['文本'])
        audio, sr = librosa.load(rec['音频路径'], sr=self.processor.feature_extractor.sampling_rate)
        return audio, text


class FRDataset():
    def get_input(self, rec, index):
        text = self.ppt.format(rec['语言'], rec['文本'])
        audio, sr = librosa.load(rec['音频路径'], sr=self.processor.feature_extractor.sampling_rate)
        return audio, text



class TrainerMix():
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        logger.info('model saved to %s', output_dir)
        model = self.accelerator.unwrap_model(self.model_wrapped)
        if self.args.save_lm_head:
            if hasattr(model, 'lm_head'):
                torch.save(model.lm_head.state_dict(), f"{output_dir}/lm_head.pt")
            else:
                torch.save(model.base_model.model.lm_head.state_dict(), f"{output_dir}/lm_head.pt")

    def log(self, logs, start_time=None):
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if 'elapsed' not in logs:
            logs['elapsed'] = time.time()-self.custom_start_time
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.args.use_adam_mini:
            from adam_mini import Adam_mini
            opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
            self.optimizer = Adam_mini(named_parameters=opt_model.named_parameters(),
                                       lr=self.args.learning_rate,
                                       betas=(self.args.adam_beta1, self.args.adam_beta2),
                                       weight_decay=self.args.weight_decay,
                                       model_sharding=False,
                                       dim=opt_model.config.hidden_size,
                                       n_heads=opt_model.config.num_attention_heads,
                                       n_kv_heads=opt_model.config.num_key_value_heads,
                                       verbose=False,
                                       )
            if is_sagemaker_mp_enabled():
                self.optimizer = smp.DistributedOptimizer(self.optimizer)
        else:
            self.optimizer = super().create_optimizer()

        return self.optimizer



class Trainer(TrainerMix, HFTrainer):
    def __init__( self, model=None, args=None, **kwargs ):
        self.custom_start_time = time.time()
        self.curr_train_step = 0
        super().__init__(model=model, args=args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None
        if self.args.use_sampler>0:
            return Sampler(self.args, 'train', train_dataset)
        else:
            return RandomSampler(self.train_dataset)


@dataclass
class TrainingArguments(HFTrainingArguments):
    rdrop: float = field(default=0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    semi_ratio: float = field(default=0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    temp: float = field(default=1, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    is_classify: bool = field(default=False, metadata={"help": "Whether to run training."})
    is_rm: bool = field(default=False, metadata={"help": "Whether to run training."})
    is_rmp: bool = field(default=False, metadata={"help": "Whether to run training."})
    save_lm_head: bool = field(default=False, metadata={"help": "Whether to run training."})

    use_adam_mini: bool = field(default=False, metadata={"help": "Whether to run training."})
    use_badam: bool = field(default=False, metadata={"help": "Whether to run training."})
    use_sampler: bool = field(default=False, metadata={"help": "Whether to run training."})
    switch_block_every: int = field(default=32, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."})
    hard_ratio: float = field( default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})



def load_unsloth_model(args, model_id):
    seed = args.unsloth_seed or args.seed
    kwargs = dict()
    logger.info('modelid %s', model_id)
    model, tokenizer = FastLanguageModel.from_pretrained(model_id, dtype=getattr(torch, args.torch_dtype), use_cache=False, max_seq_length=args.max_seq_len+8,
                                                         load_in_4bit=args.use_4bit, full_finetuning=args.use_full, load_in_8bit=args.use_8bit,
                                                         use_gradient_checkpointing='unsloth' if args.gradient_checkpointing else False, local_files_only=args.use_local, **kwargs)
    if args.use_lora:
        lora_init = args.lora_init or True
        target_modules = find_all_linear_names(args, model)
        model = FastLanguageModel.get_peft_model(model, r=args.lora_rank, lora_alpha=args.lora_alpha,
                                             lora_dropout=args.lora_dropout, bias="none",
                                             random_state=seed,
                                             use_gradient_checkpointing='unsloth' if args.gradient_checkpointing else False,
                                             target_modules=target_modules if args.lora_modules is None else args.lora_modules,
                                             use_dora=args.use_dora, init_lora_weights=lora_init)
    return model, tokenizer

def find_all_linear_names(args, model):
    import bitsandbytes as bnb
    cls = bnb.nn.Linear4bit if args.use_4bit else (bnb.nn.Linear8bitLt if args.use_8bit else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    logger.info('find linear:%s', lora_module_names)


    return list(lora_module_names)

def load_model(args, model_id):
    model_id = util.get_modelid(model_id)
    if args.use_unsloth:
        model, tokenizer = load_unsloth_model(args, model_id)

    elif args.use_lora:
        lora_init = args.lora_init or True
        from peft import LoraConfig, get_peft_model
        import bitsandbytes as bnb
        if args.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=args.use_4bit,
                load_in_8bit=args.use_8bit,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=getattr(torch, args.torch_dtype),
                bnb_4bit_use_double_quant=args.use_double_quant,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None
        kwargs = dict()
        if args.model_name.startswith('Trans'):
            model_cls = AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        else:
            model_cls = AutoModelForSeq2SeqLM
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            tokenizer = processor.tokenizer
        model = model_cls.from_pretrained(model_id, device_map={"": 0}, trust_remote_code=True, quantization_config=quantization_config,
                            torch_dtype=getattr(torch, args.torch_dtype), **kwargs)
        logger.info('model type:%s', type(model))
        task_type = 'CAUSAL_LM'
        if args.use_4bit:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        if 'KF' in model_id and hasattr(model, 'peft_config'):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_id, is_trainable=True)
        else:
            target_modules = find_all_linear_names(args, model)
            #layers_to_transform = [i for i in range(model.config.num_hidden_layers) if i >= args.lora_start_layer]
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules if args.lora_modules is None else args.lora_modules,
                #layers_to_transform=layers_to_transform,
                lora_dropout=args.lora_dropout,
                use_dora=args.use_dora,
                bias="none",
                task_type=task_type,
                #modules_to_save=args.modules_to_save,
                init_lora_weights=lora_init,
            )

            model = get_peft_model(model, lora_config)

    else:
        if args.model_name.startswith('Trans'):
            model_cls = AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        else:
            model_cls = AutoModelForSeq2SeqLM
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            tokenizer = processor.tokenizer
        kwargs = dict()
        model = model_cls.from_pretrained(model_id, torch_dtype=getattr(torch, args.torch_dtype), trust_remote_code=True, **kwargs)
        print(model)
        model = model.cuda()
    logger.info('pad token id:%s', tokenizer.pad_token_id)

    return model, tokenizer


def load_model_for_predict(args, modelid):
    torch_dtype = getattr(torch, args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(modelid, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(modelid, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map="cuda")

    logger.info('num of params for %s is %s', modelid, util.get_num_of_paras(model))
    return model, tokenizer


def setup_training(args, model, tokenizer, train_dataset, val_dataset, output_dir):
    training_args = TrainingArguments(
        remove_unused_columns=False,
        output_dir=output_dir,
        seed=args.seed+args.kfid,
        num_train_epochs=args.epochs,
        max_steps=-1,
        dataloader_num_workers=args.n_dl_worker,
        torch_compile=args.compile_model,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        #optim='adamw_torch',
        optim=args.optim,
        optim_target_modules=args.optim_target_modules if 'apollo_adamw' not in args.optim else [r".*.attn.*", r".*.mlp.*"],
        use_sampler=args.use_sampler,
        learning_rate=args.lr,
        lr_scheduler_type='linear',
        lr_scheduler_kwargs=args.lr_scheduler_paras,
        warmup_ratio=args.lr_warmup_ratio,
        warmup_steps=0,
        weight_decay=1e-2,
        max_grad_norm=args.max_grad_norm,
        logging_dir=output_dir,
        logging_steps=args.verbose,
        report_to=args.report_to,
        eval_strategy='epoch',
        eval_steps=args.eval_steps,
        eval_delay=args.eval_delay,
        save_strategy=args.save_strategy,
        save_total_limit=args.n_keep_save,
        save_steps=args.eval_steps,
        save_only_model=not args.save_opt,
        load_best_model_at_end=True if args.do_val else False,
        bf16=args.mixed_precision=='bf16',
        fp16=args.mixed_precision=='fp16',
        do_train=args.do_train,
        do_eval=args.do_val,
        do_predict=args.predict_val,
        metric_for_best_model='loss',
        disable_tqdm=True,

        ##
        hard_ratio=args.hard_ratio,
        use_adam_mini=args.use_adam_mini,
    )

    # TRAIN
    cls = Trainer
    logger.info('cls for trainer:%s', cls)
    trainer = cls(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_dataset.collate,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.early_stopping_patience > 0 and args.do_val else None,

    )
    return trainer

def prepare_dataset(args, **kwargs):
    train_data, val_data, test_data, train_ds, val_ds, test_ds = None, None, None, None, None, None
    if args.do_train or args.do_val:
        data = util.load_data(args)
        kf = KFold(n_splits=args.kn, shuffle=True, random_state=args.data_seed)
        #kf = StratifiedKFold(n_splits=args.kn, shuffle=True, random_state=args.data_seed)
        if args.groupfy:
            gps = np.array(sorted(data[args.group_col].unique()))
            splits = kf.split(gps)
            for i in range(args.kn):
                train_inds, val_inds = next(splits)
                if i==args.kfid:
                    break
            train_gp = gps[train_inds]
            val_gp = gps[val_inds]
            train_data = data[data[args.group_col].isin(train_gp)]
            val_data = data[data[args.group_col].isin(val_gp)]
        else:
            splits = kf.split(data, data.src)
            for i in range(args.kn):
                train_inds, val_inds = next(splits)
                if i==args.kfid:
                    break
            train_data = data.iloc[train_inds]
            val_data = data.iloc[val_inds]
        if args.no_validate:
            train_data = pd.concat([train_data, val_data])
            val_data = val_data[:10]

        train_ds = gen_ds(args, 'train', train_data, **kwargs)
        val_ds = gen_ds(args, 'val', val_data, **kwargs)
        logger.info('train ds:%s, val_ds:%s', len(train_ds), len(val_ds))
    if args.do_test:
        test_args = deepcopy(args)
        test_args.data_type = 'test'
        test_data = util.load_data(test_args)
        test_ds = gen_ds(args, 'test', test_data, **kwargs)
    return train_data, val_data, test_data, train_ds, val_ds, test_ds


@torch.no_grad()
def eval_trans(args, model, tokenizer, output_dir):
    train_data, val_data, test_data, train_ds, val_ds, test_ds = prepare_dataset(args, tokenizer=tokenizer)
    if args.do_val:
        df, ds = val_data, val_ds
    elif args.do_test:
        df, ds = test_data, test_ds
    elif args.do_train:
        df, ds = train_data, train_ds
    else:
        raise NotImplementedError('notme')

    rsts = []
    for i, batch in tqdm(enumerate(ds), total=len(ds)):
        input_ids, attention_masks = batch['input_ids'].cuda(), batch['attention_masks'].cuda()
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_masks=attention_masks,
            max_new_tokens=128,
            temperature=args.temp,
            num_beams=args.n_beam,
            num_return_sequences=1,
            early_stopping=True,
        )
        for gen_ids, input_ids, ID in zip(generated_ids, input_ids, batch['ID']):
            output_ids = gen_ids[len(input_ids):].tolist()
            result = tokenizer.decode(output_ids, skip_special_tokens=True)
            rsts.append([ID, result])

    preds = pd.DataFrame(rsts, columns=['ID', 'answer'])

    fpath = f"{output_dir}/pred{args.suffix}_{args.data_type}.dump"
    util.dump(preds, fpath)
    logger.info('pred saved to: %s', fpath)
    if args.do_eval:
        s = util.score(preds)
        logger.info('score is: %s', s)
    if 'answer' in df.columns:
        del df['answer']
    df['语音识别结果'] = df['中文']
    preds = preds.merge(df, on='ID')
    preds = preds[["编号", "语言", "answer", "语音识别结果", "音频路径", "ID"]]
    return preds


def restore_args(args, output_dir):
    restore_args = util.load_json(f"{output_dir}/args.json")
    for k in ['seed', 'data_seed', 'is_classify', 'is_rm', 'is_rmp', 'ds_cls', 'val_ds_cls', 'test_ds_cls', 'groupfy', 'group_col']:
        v = getattr(args, k)
        if not v:
            v = restore_args.get(k, None)
            setattr(args, k, v)
            logger.info("restored args:%s, %s", k, v)
    return args


def predict(args):
    logger.info("model:%s", args.model_name)
    args.is_eval = True
    output_dir = f"{args.data_dir}/{args.model_name}_KF{args.kfid}"
    if args.restore:
        args = restore_args(args, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if args.restore:
        if args.restore_step is not None:
            ckpt_dir = f"{output_dir}/checkpoint-{args.restore_step}"
        else:
            ckpt_dir = sorted(glob(f"{output_dir}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))[-1]
    else:
        ckpt_dir = args.backbone
    logger.info('restore from ckpt:%s', ckpt_dir)
    model, tokenizer = load_model_for_predict(args, ckpt_dir)

    if args.model_name.startswith('Trans'):
        return eval_trans(args, model, tokenizer, output_dir)


def main(args):
    logger.info('backbone: %s, kfid: %s', args.backbone, args.kfid)
    set_seed(args.seed)

    model, tokenizer = load_model(args, args.backbone)
    tokenizer.padding_side = 'right'
    output_dir = f"{args.output_dir}/{args.model_name}_KF{args.kfid}"
    os.makedirs(output_dir, exist_ok=True)
    util.dump_json(args.__dict__, f'{output_dir}/args.json')

    logger.info('num of params %s', util.get_num_of_paras(model))

    train_data, val_data, test_data, train_ds, val_ds, test_ds = prepare_dataset(args, tokenizer=tokenizer, model_config=model.config)
    trainer = setup_training(args, model, tokenizer, train_ds, val_ds, output_dir)
    if args.do_train:
        trainer.train()
        logger.info('train DONE!')
    if not args.do_train and args.do_val:
        trainer.evaluate()
        logger.info('eval DONE!')
    if args.do_test:
        outputs = trainer.predict(test_ds)
        util.dump(outputs, f'{args.output_dir}/{args.model_name}/pred_test.dump')
        print(outputs.keys())
        logger.info('test DONE!')
    logger.info('DONE!')

if __name__ == "__main__":
    util.set_logger()
    if args.debug:
        args.backbone = 'HuggingFaceTB/SmolLM-135M'
        args.num_train_epochs = 2
        args.num = 1000000
        args.eval_steps = 10
        args.batch_size = 1
        args.val_batch_size = 1
        args.dataloader_num_workers = 0
        args.gradient_accumulation_steps = 1
        args.do_train = True
        args.seed = 9528
        args.kn = 2
        args.use_full = True
        args.disable_tqdm = True
        args.ds_cls = 'Dataset'
        args.val_ds_cls = 'Dataset'
        args.ds_cls = 'VGDataset'
        args.val_ds_cls = 'VGDataset'
        #args.max_seq_len = 8
        args.max_seq_len = 4096
        args.max_gen_len = 1024
        args.n_ctx = 4

    if args.method_name is not None:
        preds = []
        for kfid in args.kfids.split():
            args.kfid = int(kfid)
            pred = globals()[args.method_name](args)
            preds.append(pred)
        if args.do_eval:
            preds = pd.concat(preds)
            s = util.score(preds)
            logger.info('kf score:%s', s)
    else:
        for kfid in args.kfids.split():
            my_args = deepcopy(args)
            my_args.kfid = int(kfid)
            my_args.seed = my_args.seed + my_args.kfid
            main(my_args)

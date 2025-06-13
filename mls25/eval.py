import os, sys, logging
import json
from glob import glob
import pandas as pd
from copy import deepcopy
import torch
import numpy as np
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback, StaticCache
from trainer import Trainer, TrainingArguments
#import trl
#from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoModelForSequenceClassification
from transformers import StoppingCriteria, StopStringCriteria, StoppingCriteriaList
from sklearn.model_selection import train_test_split, GroupKFold, KFold, StratifiedKFold
from collections import defaultdict

import util
from dataset import gen_ds
#import warnings
#warnings.filterwarnings('always')
from asr import llm_trans


sys.path.insert(0, '../')


logger = logging.getLogger(__name__)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            last_token = input_ids[0][-len(stop):]
            if torch.all(torch.eq(stop, last_token)):
                return True
        return False


def load_model(args, modelid):
    torch_dtype = getattr(torch, args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(modelid, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(modelid, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map="cuda")

    logger.info('num of params for %s is %s', modelid, util.get_num_of_paras(model))
    return model, tokenizer


def llm_trans(args, df, trans_model):
    torch_dtype = getattr(torch, args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(trans_model, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(trans_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map="cuda")

    logger.info('num of params for %s is %s', trans_model, util.get_num_of_paras(model))
    prompt = "请将下面的中文电视剧对话翻译成{}：{}"

    results = dict()
    df['l'] = df['语音识别结果'].apply(len)
    recs = df.sort_values('l', ascending=False).to_records(index=False)
    ids = range(0, len(recs), args.trans_bs)
    for s in tqdm(ids, desc='llm_trans', total=len(ids)):
        ppts = []
        batch_recs = recs[s:s+args.batch_size]
        for rec in batch_recs:
            text = rec['语音识别结果']
            language = rec['语言']
            messages = [
                {"role": "user", "content": prompt.format(language, text)}
            ]
            ppt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
            )
            ppts.append(ppt)
        model_inputs = tokenizer(ppts, return_tensors="pt", padding=True).to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            temperature=args.temp,
            num_beams=args.n_beam,
            num_return_sequences=1,
            early_stopping=True,
        )
        for gen_ids, input_ids, rec in zip(generated_ids, model_inputs.input_ids, batch_recs):
            output_ids = gen_ids[len(input_ids):].tolist()
            result = tokenizer.decode(output_ids, skip_special_tokens=True)
            results[rec["编号"]] = result
    df['answer'] = df['编号'].map(results)
    del df['l']
    return df


def prepare_dataset(args, **kwargs):
    train_ds, val_ds, test_ds = None, None, None
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
        return val_data, val_ds
    if args.do_test:
        test_args = deepcopy(args)
        test_args.data_type = 'test'
        test_data = util.load_data(test_args)
        test_ds = gen_ds(args, 'test', test_data, **kwargs)
        return test_data, test_ds




def eval_vllm(args):
    from vllm import LLM, SamplingParams
    output_dir = f"{args.output_dir}/{args.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    model = LLM(model=args.backbone,
          dtype=args.torch_dtype,
          enforce_eager=False,
          gpu_memory_utilization=0.9,
          #swap_space=4,
          #kv_cache_dtype="fp8_e5m2",
          tensor_parallel_size=1,
          trust_remote_code=True,
          max_model_len=8192,
          stops=[']]']

          #worker_use_ray=True,
         )
    tokenizer = model.get_tokenizer()

    ds = prepare_dataset(args, tokenizer=tokenizer)
    preds = []
    for batch in enumerate(ds):
        texts = batch['texts']


def restore_args(args, output_dir):
    restore_args = util.load_json(f"{output_dir}/args.json")
    for k in ['seed', 'data_seed', 'is_classify', 'is_rm', 'is_rmp', 'ds_cls', 'val_ds_cls', 'test_ds_cls', 'groupfy', 'group_col']:
        v = getattr(args, k)
        if not v:
            v = restore_args.get(k, None)
            setattr(args, k, v)
            logger.info("restored args:%s, %s", k, v)
    return args


@torch.no_grad()
def eval_trans(args, model, tokenizer, output_dir):
    df, ds = prepare_dataset(args, tokenizer=tokenizer)

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


def main(args):
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
    model, tokenizer = load_model(args, ckpt_dir)

    if args.model_name.startswith('Trans'):
        return eval_trans(args, model, tokenizer, output_dir)



if __name__ == "__main__":
    args = util.parser.parse_args()
    util.set_logger()
    if args.debug:
        args.backbone = 'HuggingFaceTB/SmolLM-135M'
        args.num_train_epochs = 2
        args.max_seq_len = 8
        args.num = 10
        args.eval_steps = 2
        args.batch_size = 1
        args.val_batch_size = 1
        args.gradient_accumulation_steps = 1
        args.do_train = True
        args.seed = 9527
        args.kn = 2
        args.use_full = True
        args.ds_cls = 'PretrainDataset'
        args.val_ds_cls = 'PretrainDataset'
    preds = []
    for kfid in args.kfids.split():
        args.kfid = int(kfid)
        pred = main(args)
        preds.append(pred)
    if args.do_eval:
        preds = pd.concat(preds)
        s = util.score(preds)
        logger.info('kf score:%s', s)


import os, sys, json, logging
from copy import deepcopy
import pickle
import pandas as pd
from tqdm import tqdm
from glob import glob
from collections import defaultdict
import time
from argparse import ArgumentParser
from multiprocessing import Pool
import numpy as np
import subprocess
from contextlib import contextmanager
import re
import hashlib
from sklearn.metrics import f1_score
import sklearn.metrics
from scipy.spatial import KDTree
import torch
from torch.nn import functional as F
import numpy as np
import librosa
from nltk.translate.bleu_score import sentence_bleu

logger = logging.getLogger(__name__)


parser = ArgumentParser(conflict_handler='resolve')
parser.add_argument("-d", "--debug",  action="store_true")
parser.add_argument("-data_type", default='train')
parser.add_argument("-nv","--no_val", action="store_true")
parser.add_argument("-is_eval",  action="store_true")
parser.add_argument("-ds", "--dataset", default='mls25')
parser.add_argument("-ds_cls")
parser.add_argument("-val_ds_cls")
parser.add_argument("-test_ds_cls")
parser.add_argument("-data_type", default='train')
parser.add_argument("-data_dir", default='../data')
parser.add_argument("-kn", type=int)
parser.add_argument("-kfid", type=int, default=0)
parser.add_argument("-kfids")
parser.add_argument("-groupfy",  action="store_true")
parser.add_argument("-group_col", default='片段名')
parser.add_argument("-model_name")
parser.add_argument("-backbone")
parser.add_argument("-activation")
parser.add_argument("-n_repeat", type=int, default=1)
parser.add_argument("-n_layer", type=int)
parser.add_argument("-d_model", type=int)
parser.add_argument("-d_ffd", type=int)
parser.add_argument("-n_head", type=int)
parser.add_argument("-dropout", type=float, default=0)
parser.add_argument("-val_pct", default=0.1)
parser.add_argument("-seed", type=int)
parser.add_argument("-data_seed", type=int)
parser.add_argument("-unsloth_seed", type=int)
parser.add_argument("-n_xtoken", type=int)
parser.add_argument("-prefix")
parser.add_argument("-max_seq_len", type=int, default=8)
parser.add_argument("-max_gen_len", type=int, default=8)
parser.add_argument("-mixed_precision", default='no')
parser.add_argument("-cpu",  action="store_true")
parser.add_argument("-cudnn_benchmark",  action="store_true")
parser.add_argument("-deterministic_algorithms",  action="store_true")
parser.add_argument("-save",  action="store_true")
parser.add_argument("-save_half",  action="store_true")
parser.add_argument("-save_state",  action="store_true")
parser.add_argument("-save_best",  action="store_true")
parser.add_argument("-ckpts", default=None)
parser.add_argument("-n_keep_save",  type=int, default=1)
parser.add_argument("-save_epoch",  type=int, default=100000000000)
parser.add_argument("-save_opt",  action="store_true")
parser.add_argument("-remove_unused_columns",  action="store_true")
parser.add_argument("-gradient_checkpointing",  action="store_true")
parser.add_argument("-max_grad_norm",  type=float, default=0)
parser.add_argument("-use_pretrain",  action="store_true")
parser.add_argument("-use_sampler",  action="store_true")
parser.add_argument("-use_badam",  action="store_true")
parser.add_argument("-use_full",  action="store_true")
parser.add_argument("-switch_block_every",  type=int, default=32)
parser.add_argument("-use_score_scaling",  action="store_true")
parser.add_argument("-torch_compile",  action="store_true")
parser.add_argument("-use_double_quant",  action="store_true")
parser.add_argument("-m", "--method_name")
parser.add_argument("-compile_model", action="store_true")
parser.add_argument("-compile_dynamic", action="store_true", help="comiple pytorch")
parser.add_argument("-compile_mode", default="default")
parser.add_argument("-torch_dtype", default="bfloat16")

# transformers
parser.add_argument("-evaluation_strategy", default='steps')
parser.add_argument("-save_strategy", default='steps')
parser.add_argument("-eval_steps", type=int, default=1000)
parser.add_argument("-eval_delay", type=int, default=0)

parser.add_argument("-bs", "--batch_size", type=int)
parser.add_argument("-mbs", "--min_batch_size", type=int)
parser.add_argument("-vbs", "--val_batch_size", type=int)
parser.add_argument("-lr", type=float, default=1e-4)
parser.add_argument("-lr_scheduler", default='ld')
parser.add_argument("-lr_scheduler_paras", type=json.loads, help='lr scheduler parameters')
parser.add_argument("-lr_warmup_ratio", type=float, default=0.0)
parser.add_argument("-lr_decay_rate", type=float, default=1.0)

parser.add_argument("-init_kl_coef", type=float, default=0.2)
parser.add_argument("-max_grad_norm", type=float, default=1)
parser.add_argument("-gas", "--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("-opt", default="torch.optim.AdamW")
parser.add_argument("-optim", default="adamw_torch")
parser.add_argument("-optim_target_modules", nargs="+", default=None)
parser.add_argument("-opt_paras", type=json.loads, default={}, help='parameters for optimizer')
parser.add_argument("-n_dl_worker", type=int, default=0)
parser.add_argument("-weight_decay", type=float, default=1e-2)
parser.add_argument("-output_dir", default="../data")
parser.add_argument("-output_name")
parser.add_argument("-norm_func")
parser.add_argument("-verbose", type=int, default=16)
parser.add_argument("-report_to", default="tensorboard")
parser.add_argument("-min_cnt", type=int, default=10)
parser.add_argument("-output_keys", nargs='+')
parser.add_argument("-val_by_score",  action="store_true")
parser.add_argument("-val_steps", type=int, default=1000)
parser.add_argument("-init_epoch", type=int, default=0)
parser.add_argument("-epochs", type=int, default=10)
parser.add_argument("-n_init_epoch", type=int, default=0)
parser.add_argument("-n_epoch_step", type=int, default=10000000000)
parser.add_argument("-n_val_epoch_step", type=int, default=10000000000)
parser.add_argument("-es", "--early_stopping_patience", type=int, default=1)
parser.add_argument("-es_min_delta", type=float, default=0.0005)
parser.add_argument("-num", type=int, default=10000000000)
parser.add_argument("-temp", type=float, default=1.0)
parser.add_argument("-ema", type=float, default=0)
parser.add_argument("-ema_start", type=float, default=0)
parser.add_argument("-topp", type=float, default=1.0)
parser.add_argument("-topk", type=int, default=-1)
parser.add_argument("-n_best", type=int, default=1)
parser.add_argument("-max_temp", type=float, default=5)
parser.add_argument("-min_temp", type=float, default=0.01)
parser.add_argument("-temp_decay_step", type=int, default=10000)
parser.add_argument("-n_frozen", type=int, default=0)
parser.add_argument("-frozen_emb",  action="store_true")
parser.add_argument("-use_lora",  action="store_true")
parser.add_argument("-lora_rank", type=int, default=32)
parser.add_argument("-lora_alpha", type=int, default=16)
parser.add_argument("-lora_dropout", type=float, default=0)
parser.add_argument("-lora_modules", nargs='+', default=None)
parser.add_argument("-lora_init")
parser.add_argument("-use_dora",  action="store_true")
parser.add_argument("-freeze_lm",  action="store_true")
parser.add_argument("-unfreeze_lm_head",  action="store_true")
parser.add_argument("-use_rslora",  action="store_true",default=None)
parser.add_argument("-ext_ratio",  type=float, default=0)
parser.add_argument("-use_orcamath",  action="store_true")
parser.add_argument("-use_mustard",  action="store_true")
parser.add_argument("-restore",  action="store_true")
parser.add_argument("-restore_model",  default="NA")
parser.add_argument("-restore_step",  type=int)
parser.add_argument("-do_train",  action="store_true")
parser.add_argument("-do_val",  action="store_true")
parser.add_argument("-do_test",  action="store_true")
parser.add_argument("-do_concat",  action="store_true", default=None)
parser.add_argument("-nt", "--no_train",  action="store_true")
parser.add_argument("-to_list",  action="store_true", default=None)
parser.add_argument("-predict_val",  action="store_true")
parser.add_argument("-scoring",  action="store_true")
parser.add_argument("-predict_test",  action="store_true")
parser.add_argument("-use_unsloth",  action="store_true")
parser.add_argument("-use_dora",  action="store_true")
parser.add_argument("-use_4bit", action="store_true")
parser.add_argument("-use_8bit", action="store_true")
parser.add_argument("-no_validate", action="store_true")
parser.add_argument("-hard_ratio", type=float, default=0)
parser.add_argument("-suffix", default="")
parser.add_argument("-split", type=int, default=-1)
parser.add_argument("-mixup", type=float, default=0)

parser.add_argument("-n_beam", type=int, default=1)
parser.add_argument("-n_ctx", type=int, default=1)
parser.add_argument("-n_max_token", type=int, default=256)
parser.add_argument("-ppt")
parser.add_argument("-trans_model")
parser.add_argument("-trans_bs", type=int, default=1)
parser.add_argument("-llm_dir")
parser.add_argument("-return_timestamps", action="store_true")
parser.add_argument("-use_adam_mini", action="store_true")
parser.add_argument("-use_gold", action="store_true")
parser.add_argument("-use_16k", action="store_true")
parser.add_argument("-use_local", action="store_true")
parser.add_argument("-is_lm", action="store_true")
parser.add_argument("-is_e2e", action="store_true")
parser.add_argument("-lan")
parser.add_argument("-aug_zh", type=float, default=0)
parser.add_argument("-min_conf", type=float, default=0)
parser.add_argument("-n_per_name", type=int, default=0)
parser.add_argument("-no_en", action="store_true")

def bleu(texts, preds):
    ss = []
    for text, pred in zip(texts, preds):
        s = sentence_bleu([text.split(" ")], pred.split(" "), weights=(0.5, 0.5))
        ss.append(s)
    s = np.mean(ss)
    return s

def score(preds, solutions, ignore_loc=False):
    preds = deepcopy(preds)
    return s




def restore_args(args, output_dir):
    restore_args = load_json(f"{output_dir}/args.json")
    ks1 = ['seed', 'data_seed', 'backbone', 'ds_cls', 'val_ds_cls', 'test_ds_cls', 'output_keys', 'ckpts', 'to_list', 'do_concat', 'norm_func', 'activation']
    ks1 += ['ks1', 'ks2', 'n_input_layer', 'input_ks1', 'input_ks2', 'flatten_c', 'is_cls', 'resblock', 'mlp_ratio', 'kernel_size', 'n_layer', 'd_model', 'n_head', 'd_ffd']
    ks2 = []
    new_args = []
    for k in (ks1+ks2):
        v = getattr(args, k)
        if v is None:
            v = restore_args.get(k, None)
            setattr(args, k, v)
            new_args.append((k, v))
    logger.info("restored args:%s", new_args)


def get_data_solution(args, data, dataset, data_type):
    return data


def get_audio_len(fpath, sr=None):
    audio, sr = load_audio(fpath, sr=sr)
    l = len(audio) / sr
    return l


def get_audio_lens(df, n_task=8):
    df2 = df.groupby('音频路径').head(1).reset_index(drop=True)
    with Pool(n_task) as p:
        rsts = []
        for rst in tqdm(p.imap(get_audio_len, df2["音频路径"], chunksize=8), total=len(df2), desc='get_audio_lens'):
            rsts.append(rst)
    df2['audio_len'] = rsts
    df = df.merge(df2[['音频路径', 'audio_len']], on='音频路径')
    return df


def load_alt(args):
    df = pd.read_csv('../data/alt.csv')
    return df


def load_data(args):
    if args.dataset=='mls25':
        if args.data_type=='train':
            df = pd.read_csv(f"{args.data_dir}/mls25/text_data/train.csv")
            df['ID'] = range(len(df))
            df = df.sort_values('音频路径').reset_index(drop=True)
            df['音频路径'] = df['音频路径'].apply(lambda x: f"{args.data_dir}/mls25{x}")
        elif args.data_type=='val':
            df = pd.read_csv(f"{args.data_dir}/mls25/text_data/dev.csv")
            df['ID'] = range(len(df))
            df = df.sort_values('音频路径').reset_index(drop=True)
            df['音频路径'] = df['音频路径'].apply(lambda x: f"{args.data_dir}/mls25{x}")
        elif args.data_type=='test':
            df = pd.read_csv(f"{args.data_dir}/mls25/text_data/testa.csv")
            df['ID'] = range(len(df))
            df = df.sort_values('音频路径').reset_index(drop=True)
            df['音频路径'] = df['音频路径'].apply(lambda x: f"{args.data_dir}/mls25{x}")
        df['name'] = df["片段名"].apply(lambda x: x.split(" EP")[0])
    elif args.dataset=='mls25_4000':
        num = args.num
        args = deepcopy(args)
        args.dataset = 'mls25'
        args.data_type = 'train'
        args.num = 10000000000
        df = load_data(args)
        df = df.sample(4000, random_state=10000)
        df = df[:num]
        return df
    elif args.dataset=='alt':
        df = pd.read_csv(f'{args.data_dir}/alt.csv')
        df = df.sample(frac=1, random_state=1000)
        if args.no_en:
            df = df[df['语言']!='英语']
    elif args.dataset=='wenet_drama':
        df = pd.read_csv(f'{args.data_dir}/wenet_drama.csv')
        if args.n_per_name>0:
            df = df.groupby('name').head(args.n_per_name)
        df = df.sample(frac=1, random_state=1000)
    else:
        raise NotImplementedError(args.dataset)
    df['src'] = args.dataset
    if args.lan is not None:
        df = df[df["语言"]==args.lan]

    df = df[:args.num]
    if args.use_16k:
        df['音频路径'] = df['音频路径'].apply(lambda x: f"../data/processed/{args.dataset}_{args.data_type}_16k/{os.path.basename(x)}")
    logger.info('num of %s is %s', args.dataset, len(df))
    return df




def load_kf_preds(args):
    preds = []
    for kfid in args.kfid.split():
        output_dir = f"{args.data_dir}/{args.model_name}_KF{args.kfid}"
        pred = load_dump(f"{output_dir}/pred{args.suffix}_{args.data_type}.dump")
        preds.append(pred)
    if args.data_type!='test':
        preds = pd.concat(preds)
    return preds


def get_modelid(model_name):
    if 'KF' in model_name:
        modelid = sorted(glob(f"{model_name}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))[-1]
    else:
        modelid = model_name
    return modelid


def set_logger(level=logging.INFO):
    logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    logger.setLevel(level)


def get_md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()

@contextmanager
def timer(name):
    t0 = time.time()
    #print('{} start'.format(name))
    logger.info('%s start', name)
    yield
    #print('{} done in {} seconds'.format(name, time.time() - t0))
    logger.info('%s done in %s seconds', name, time.time()-t0)

def load_dump(fpath):
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    return data


def dump(data, fpath, protocol=2):
    fdir = os.path.dirname(fpath)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(fpath, 'wb') as f:
        pickle.dump(data, f)


def load_json(fpath):
    with open(fpath) as f:
        dictionary = json.load(f)
    return dictionary


def load_json_lines(fpath, num=1e16, exclude_keys=None, post_process=None):
    if exclude_keys is None:
        exclude_keys = []
    data = []
    with open(fpath) as f:
        for l in f:
            dic = json.loads(l)
            for k in exclude_keys:
                if k in dic:
                    _ = dic.pop(k)
            if post_process is not None:
                dic = post_process(dic)
            data.append(dic)
            if len(data)>=num:
                break
    return data


def dump_json(dictionary, fpath, ensure_ascii=False):
    with open(fpath, 'w') as f:
        json.dump(dictionary, f, ensure_ascii=ensure_ascii)


def dump_json_lines(dicts, fpath, ensure_ascii=False):
    with open(fpath, 'w', encoding='utf8') as f:
        for d in dicts:
            json.dump(d, f, ensure_ascii=ensure_ascii)
            f.write(os.linesep)

def dynamic_import(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def timestamp():
    return time.strftime('%Y%m%d%H%M%S')


def get_num_of_paras(m):
    num1, num2 = 0, 0
    for p in m.parameters():
        if p.requires_grad:
            num1 += p.numel()
        else:
            num2 += p.numel()
    return num1/1000/1000, num2/1000/1000


def load_audio(fpath, sr=None):
    audio, sr = librosa.load(fpath, sr=sr)
    return audio, sr


def load_img(fpath, flag=None):
    if flag is None:
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(fpath, flag)
    return img




if __name__ == "__main__":
    args = parser.parse_args([])
    args.data_type = 'train'
    #df = load_data(args)
    load_gen_data(args)

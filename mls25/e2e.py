import os, sys, logging
import re
from glob import glob
from tqdm import tqdm
import inspect
from itertools import combinations
import numpy as np
import pandas as pd
import util
import math
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

import torch
import torch.distributed as dist
from pt_util import get_parameter_names


logger = logging.getLogger(__name__)


class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)

    def get_distribute_data(self, data, world_size=None, rank=None):
        rank = dist.get_rank() if rank is None else rank
        world_size = dist.get_world_size() if world_size is None else world_size
        per_rank = int(math.ceil(len(data) / world_size))
        return data[rank * per_rank:(rank + 1) * per_rank]

    def get_iter_items(self, index):
        rec = self.data[index]
        yield self.getitem(index, rec=rec)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        data = self.data
        if worker_info is not None:
            self.data = self.get_distribute_data(data, worker_info.num_workers, worker_info.id)
        for index in range(len(self.data)):
            items = self.get_iter_items(index)
            for item in items:
                if item is not None:
                    yield item

def nllb(args, df):
    torch_dtype = getattr(torch, args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.trans_model, src_lang="zho_Hans")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.trans_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    logger.info('num of params for %s is %s', args.trans_model, util.get_num_of_paras(model))

    ml_translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang="zho_Hans", tgt_lang="zsm_Latn", max_length=512, device='cuda:0')
    th_translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang="zho_Hans", tgt_lang="tha_Thai", max_length=512, device='cuda:0')
    en_translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang="zho_Hans", tgt_lang="eng_Latn", max_length=512, device='cuda:0')

    results = []
    for rec in tqdm(df.to_records(index=False), desc='nllb', total=len(df)):
        text = rec['语音识别结果']
        language = rec['语言']
        if language == '泰语':
            translator = th_translator
        elif language == '英语':
            translator = en_translator
        elif language == '马来语':
            translator = ml_translator
        else:
            raise NotImplementedError(language)
        try:
            result = translator(text)[0]
            result = result['translation_text']
        except Exception as e:
            logger.error(f"text:{text}, error:{e}, rec:{rec}")
            result = ''
        results.append(result)
    df['answer'] = results


def asr_fr(args, df):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(torch, args.torch_dtype)
    from fireredasr.models.fireredasr import FireRedAsr
    if 'aed' in args.model_name.lower():
        mt = 'aed'
        model = FireRedAsr.from_pretrained(mt, args.model_name)
    else:
        mt = 'llm'
        model = FireRedAsr.from_pretrained(mt, args.model_name, llm_dir=args.llm_dir)
    fpaths = df['音频路径'].values.tolist()
    IDs = df['ID'].values.tolist()
    num = (len(fpaths)+args.batch_size-1)//args.batch_size
    asrs = dict()
    for i in tqdm(range(len(df['音频路径'])), desc="asr", total=num):
        batch_ID = IDs[i*args.batch_size:(i+1)*args.batch_size]
        batch_fpath = fpaths[i*args.batch_size:(i+1)*args.batch_size]
        asr_results = model.transcribe(batch_ID, batch_fpath, dict(use_gpu=1, beam_size=args.n_beam, nbest=1, decode_max_len=0, softmax_smoothing=1.25, aed_length_penalty=0.6, eos_penalty=1.0))
        for asr_result in asr_results:
            asrs[asr_result['uttid']] = asr_result['text']
    df['语音识别结果'] = df['ID'].map(asrs)


def asr(args, df):
    if 'firered' in args.model_name.lower():
        return asr_fr(args, df)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model_id = args.model_name

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    logger.info('num of params for %s is %s', args.model_name, util.get_num_of_paras(model))

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=args.return_timestamps,
    )
    asrs = []
    for fpath in tqdm(df['音频路径'], desc="asr", total=len(df)):
        asr_result = pipe(fpath)
        asrs.append(asr_result['text'])
    #fpaths = df['音频路径'].values
    #num = (len(fpaths)+args.batch_size-1)//args.batch_size
    #for i in tqdm(range(len(df['音频路径'])), desc="asr", total=num):
        #asr_results = pipe(fpaths[i:i+args.batch_size])
        #for asr_result in asr_results:
        #    asrs.append(asr_result['text'])
    df['语音识别结果'] = asrs




def main(args):
    df = util.load_data(args)
    output_fpath = f'../data/{args.output_name}.csv'
    if not os.path.exists(output_fpath):
        asr(args, df)
        df.to_csv(output_fpath, index=False)
    else:
        logger.info(f'ignore asr for file existed:{output_fpath}')
    df = pd.read_csv(output_fpath)
    nllb(args, df)
    df = df.sort_values(['ID']).reset_index(drop=True)
    df = df[["编号", "answer", "语音识别结果"]]
    df.to_csv(output_fpath, index=False)
    logger.info(f'Done! Saved:{output_fpath}')


if __name__ == "__main__":
    args = util.parser.parse_args()
    gl = globals()
    if args.debug:
        util.set_logger(logging.DEBUG)
    else:
        util.set_logger()
    main(args)

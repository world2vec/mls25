import os, sys, logging
os.environ['MODELSCOPE_LOG_LEVEL'] = '30' # Set to WARNING
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

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import transformers
transformers.logging.set_verbosity_error()

import torch
import torch.distributed as dist
from pt_util import get_parameter_names
import librosa


logger = logging.getLogger(__name__)

trans_ppt = "请将下面的中文电视剧对话翻译成{}：{}"
trans_ppt2 = "请将下面的{}翻译成{}：{}"
qw2audio_asrppt = "<|audio_bos|><|AUDIO|><|audio_eos|>" + "Detect the language and recognize the speech: <|zh|>{}<|endoftext|>"
qw2audio_sttppt = "<|audio_bos|><|AUDIO|><|audio_eos|>" + "Detect the language and translate the speech into {}: <|zh|>{}<|endoftext|>"
lanencodes = {"马来语":"Malay", "英语":"English", "泰语":"Thai"}


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
    return df

def get_modelid(model_name):
    if 'KF' in model_name:
        modelid = sorted(glob(f"{model_name}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))[-1]
    else:
        modelid = model_name
    return modelid


def llm_trans_ctx(args, df):
    torch_dtype = getattr(torch, args.torch_dtype)
    trans_model = get_modelid(args.trans_model)
    tokenizer = AutoTokenizer.from_pretrained(trans_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(trans_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map="auto")

    logger.info('num of params for %s is %s', trans_model, util.get_num_of_paras(model))
    prompt = '''请将下面的中文电视剧对话翻译成{}，请保留对话分割符"|"：'''
    df = df.sort_values(["音频路径"]).reset_index(drop=True)
    results = dict()
    for (name, lan), gp in tqdm(df.groupby(["name", "语言"]), desc="llm_trans_ctx"):
        recs = gp.to_records(index=False)
        last, num, n_token = 0, 0, 0
        ID2tokens, ID2text = dict(), dict()
        while last<(len(recs)):
            text = recs[last]['语音识别结果']
            ctx_texts = [text]
            ctx_recs = [recs[last]]
            tokens = tokenizer.tokenize(text)
            n_token = len(tokens)
            ID = recs[last]['ID']
            ID2tokens[ID] = tokens
            ID2tokens[ID] = text
            for i in range(args.n_ctx):
                if (last-i-1)<0:
                    break
                rec = recs[last-i-1]
                ID = rec['ID']
                tokens = ID2tokens[ID]
                text = ID2text[ID]
                if n_token+len(tokens)<args.n_max_token:
                    ctx_texts = [text] + ctx_texts
                    ctx_recs = [rec] + ctx_recs
                    n_token += len(tokens)
            n_ctx = len(ctx_texts) - 1
            last += 1

            for i in range(last, len(recs)):
                rec = recs[i]
                text = rec['语音识别结果']
                tokens = tokenizer.tokenize(text)
                if (n_token + len(tokens))>args.n_max_token:
                    break
                else:
                    ctx_texts.append(text)
                    ctx_recs.append(rec)
                    n_token += len(tokens)
                    last += 1
            text = "|".join(ctx_texts)
            messages = [
                {"role": "user", "content": prompt.format(lan, text)}
            ]
            ppt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
            )
            model_inputs = tokenizer([ppt], return_tensors="pt", padding=True).to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=args.n_max_token*2,
                temperature=args.temp,
                num_beams=args.n_beam,
                num_return_sequences=1,
                early_stopping=True,
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            outputs = tokenizer.decode(output_ids, skip_special_tokens=True)
            outputs = outputs.split("|")
            assert len(outputs) == len(ctx_texts), (ctx_texts, outputs)
            for output, rec in zip(outputs[n_ctx:], ctx_recs[n_ctx:]):
                results[rec["编号"]] = output


def llm_trans(args, df):
    if args.n_ctx>1:
        return llm_trans_ctx(args, df)
    torch_dtype = getattr(torch, args.torch_dtype)
    trans_model = get_modelid(args.trans_model)
    tokenizer = AutoTokenizer.from_pretrained(trans_model, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(trans_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map="cuda")

    logger.info('num of params for %s is %s', trans_model, util.get_num_of_paras(model))
    prompt = globals()[args.ppt]
    #prompt = "请将下面的中文电视剧对话翻译成{}：{}"
    logger.info('use ppt:%s', prompt)

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
            if args.ppt=='trans_ppt':
                messages = [
                    {"role": "user", "content": prompt.format(language, text)}
                ]
            elif args.ppt=='trans_ppt2':
                messages = [
                    {"role": "user", "content": prompt.format('中文', language, text)}
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

def asr_dol(args, df):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(torch, args.torch_dtype)
    modelid = get_modelid(args.model_name)
    import dolphin
    model = dolphin.load_model("small", modelid, "cuda")
    fpaths = df['音频路径'].values.tolist()
    IDs = df['ID'].values.tolist()
    num = (len(fpaths)+args.batch_size-1)//args.batch_size
    asrs = dict()
    for i in tqdm(range(num), desc="asr", total=num):
        fpath = fpaths[i]
        ID = IDs[i]
        waveform = dolphin.load_audio(fpath)
        asr_result = model(waveform, lang_sym="zh").text_nospecial
        asrs[ID] = asr_result
    df['语音识别结果'] = df['ID'].map(asrs)
    return df

def asr_fr(args, df):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(torch, args.torch_dtype)
    from fireredasr.models.fireredasr import FireRedAsr
    if 'aed' in args.model_name.lower():
        mt = 'aed'
        model = FireRedAsr.from_pretrained(mt, args.model_name)
    else:
        mt = 'llm'
        model = FireRedAsr.from_pretrained(mt, args.model_name, llm_dir=args.llm_dir, torch_dtype=args.torch_dtype)
    fpaths = df['音频路径'].values.tolist()
    IDs = df['ID'].values.tolist()
    num = (len(fpaths)+args.batch_size-1)//args.batch_size
    asrs = dict()
    for i in tqdm(range(num), desc="asr", total=num):
        batch_ID = IDs[i*args.batch_size:(i+1)*args.batch_size]
        batch_fpath = fpaths[i*args.batch_size:(i+1)*args.batch_size]
        if mt=='aed':
            asr_results = model.transcribe(batch_ID, batch_fpath, dict(use_gpu=1, beam_size=args.n_beam, nbest=1, decode_max_len=0, softmax_smoothing=1.25, aed_length_penalty=0.6, eos_penalty=1.0))
        else:
            asr_results = model.transcribe(batch_ID, batch_fpath,
                                           dict(use_gpu=1, beam_size=args.n_beam, decode_max_len=0, decode_min_len=0, repetition_penalty=3.0, llm_length_penalty=1.0, temperature=1.0))
        for asr_result in asr_results:
            asrs[asr_result['uttid']] = asr_result['text']
    df['语音识别结果'] = df['ID'].map(asrs)
    return df


def asr_ms(args, df):
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks


    model_id = args.model_name
    pipe = pipeline(
        task=Tasks.auto_speech_recognition,
        model=model_id,
        model_revision="v2.0.4",
    )

    asrs = []
    for fpath in tqdm(df['音频路径'], desc="asr", total=len(df)):
        asr_result = pipe(fpath)[0]
        asrs.append(asr_result['text'])
    df['语音识别结果'] = asrs


def _asr(model, processor, fpath, text):
    audio, sr = librosa.load(fpath, sr=processor.feature_extractor.sampling_rate)
    inputs = processor(text=text, audios=audio, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.cuda()

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

def asr_lm(args, df):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    lanencodes = {"马来语": "Malay", "英语": "English", "泰语": "Thai"}

    modelid = get_modelid(args.model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        modelid, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=True
    )
    model.to(device)
    logger.info('num of params for %s is %s', args.model_name, util.get_num_of_paras(model))

    try:
        processor = AutoProcessor.from_pretrained(modelid, trust_remote_code=True)
    except:
        #processor = AutoProcessor.from_pretrained('Qwen/Qwen2-Audio-7B', trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model.peft_config['default'].base_model_name_or_path, trust_remote_code=True)

    prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>" + "Detect the language and recognize the speech: <|zh|>"

    asrs = []
    for rec in tqdm(df.to_records(index=False), desc="asr", total=len(df)):
        asr_result = _asr(model, processor, rec["音频路径"], prompt)
        asrs.append(asr_result)
    # fpaths = df['音频路径'].values
    # num = (len(fpaths)+args.batch_size-1)//args.batch_size
    # for i in tqdm(range(len(df['音频路径'])), desc="asr", total=num):
    # asr_results = pipe(fpaths[i:i+args.batch_size])
    # for asr_result in asr_results:
    #    asrs.append(asr_result['text'])
    df['语音识别结果'] = asrs
    return df


def e2e(args, df):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    lanencodes = {"马来语":"Malay", "英语":"English", "泰语":"Thai"}

    modelid = get_modelid(args.model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        modelid, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=True
    )
    model.to(device)
    logger.info('num of params for %s is %s', args.model_name, util.get_num_of_paras(model))

    try:
        processor = AutoProcessor.from_pretrained(modelid, trust_remote_code=True)
    except:
        #processor = AutoProcessor.from_pretrained('Qwen/Qwen2-Audio-7B', trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model.peft_config['default'].base_model_name_or_path, trust_remote_code=True)

    prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>" + "Detect the language and translate the speech into {}: <|zh|>"


    asrs = []
    for rec in tqdm(df.to_records(index=False), desc="asr", total=len(df)):
        text = prompt.format(lanencodes[rec['语言']])
        asr_result = _asr(model, processor, rec["音频路径"], text)
        asrs.append(asr_result)
    #fpaths = df['音频路径'].values
    #num = (len(fpaths)+args.batch_size-1)//args.batch_size
    #for i in tqdm(range(len(df['音频路径'])), desc="asr", total=num):
        #asr_results = pipe(fpaths[i:i+args.batch_size])
        #for asr_result in asr_results:
        #    asrs.append(asr_result['text'])
    df['answer'] = asrs
    return df


def asr(args, df):
    if args.batch_size>1:
        df = util.get_audio_lens(df)
        df = df.sort_values(['audio_len'], ascending=False)
    if 'firered' in args.model_name.lower():
        return asr_fr(args, df)
    elif 'iic' in args.model_name:
        return asr_ms(args, df)
    elif 'dolphin' in args.model_name:
        return asr_dol(args, df)
    elif args.is_lm:
        return asr_lm(args, df)
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
    return df


def main(args):
    df = util.load_data(args)
    output_fpath = f'../data/{args.output_name}.csv'
    if args.is_e2e:
        df = e2e(args, df)
        df['语音识别结果'] = None
    else:
        if args.use_gold:
            df['语音识别结果'] = df['中文']
            df.to_csv(output_fpath, index=False)
        elif not os.path.exists(output_fpath):
            df2 = df.groupby(['音频路径']).head(1).reset_index(drop=True)
            df2 = asr(args, df2)
            df = df.merge(df2[["音频路径", "语音识别结果"]])
            df.to_csv(output_fpath, index=False)
        else:
            logger.info(f'ignore asr for file existed:{output_fpath}')
        df = pd.read_csv(output_fpath)
        if 'nllb' in args.trans_model.lower():
            df = nllb(args, df)
        else:
            df = llm_trans(args, df)
    df = df.sort_values(['ID']).reset_index(drop=True)
    if args.data_type!='test':
        logger.info('bleu:%s', util.bleu(df["文本"], df["answer"]))
    df = df[["编号", "语言", "answer", "语音识别结果", "音频路径", "ID"]]
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

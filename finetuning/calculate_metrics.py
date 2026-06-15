import os
import re
import json
import math
import torch
import torchaudio
import argparse
import unicodedata
import pandas as pd
import numpy as np
import pyworld as pw

import utmosv2

from tqdm import tqdm
from jiwer import cer
from concurrent.futures import ProcessPoolExecutor, as_completed
from pymcd.mcd import Calculate_MCD
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

import warnings
warnings.filterwarnings("ignore")

MOS_MODEL = utmosv2.create_model(pretrained=True) # Default sr: 16000
ASR_MODEL = ASRInferencePipeline(model_card="omniASR_CTC_1B") # Default sr: 16000    
MCD_MODEL = Calculate_MCD(MCD_mode='dtw') # Default sr: 22050
TARGET_RATE = 16000

LANG_TO_OMNIASR = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "af": "afr_Latn",
    "jv": "jav_Latn",
    "km": "khm_Khmr",
    "np": "nep_Deva",
    "si": "sin_Sinh",
    "su": "sun_Latn",
    "tn": "tsn_Latn",
    "xh": "xho_Latn",
}

CKPT_STEP_PER_LANG = {
    'en': 300000,
    'de': 300000,
    'zh': 300000,
    'af': 100000,
    'tn': 100000,
    'xh': 100000,
    'si': 100000,
    'jv': 100000,
    'su': 100000,
    'km': 100000,
    'np': 100000,
}

def normalize_text(text):
    
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\w\s]', '', text).lower()
    
    return text

def load_and_resample(wav_path):
    y, sr = torchaudio.load(wav_path)
    if sr != TARGET_RATE:
        y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=TARGET_RATE)
    return y

def calculate_mcd(gt_wav_path, gen_wav_path):
    
    mcd = MCD_MODEL.calculate_mcd(gt_wav_path, gen_wav_path)
    
    return mcd

def calculate_f0_error(gt_wav_path, gen_wav_path, log_metric=True):
    
    gt_wav, sr = torchaudio.load(gt_wav_path)
    gen_wav, sr = torchaudio.load(gen_wav_path)
    gt_wav = gt_wav.numpy().squeeze().astype(np.float64)
    gen_wav = gen_wav.numpy().squeeze().astype(np.float64)
    gt_f0, gt_time = pw.harvest(gt_wav, sr)
    gen_f0, gen_time = pw.harvest(gen_wav, sr)
    
    distance, path = fastdtw(gt_f0.reshape(-1, 1), gen_f0.reshape(-1, 1), dist=euclidean)
    
    path = np.array(path)
    gt_f0_aligned = gt_f0[path[:, 0]]
    gen_f0_aligned = gen_f0[path[:, 1]]
    mask = (gt_f0_aligned > 0) & (gen_f0_aligned > 0)
    
    if not np.any(mask):
        return 0.0

    gt_voiced = gt_f0_aligned[mask]
    gen_voiced = gen_f0_aligned[mask]
    
    if log_metric:
        gt_voiced = np.log(gt_voiced)
        gen_voiced = np.log(gen_voiced)

    f0_rmse = np.sqrt(np.mean((gt_voiced - gen_voiced) ** 2))
    
    return f0_rmse

def _mcd_worker(idx, pair):
    gt, gen = pair
    mcd = calculate_mcd(gt, gen)
    return idx, mcd

def _f0_worker(idx, pair, log_metric):
    gt, gen = pair
    f0 = calculate_f0_error(gt, gen, log_metric=log_metric)
    return idx, f0

def _unwrap_utmos(utmos_list):
    utmos_dict = {}
    for item in utmos_list:
        fpath = item['file_path']
        fid = os.path.basename(fpath).split('.')[0]
        score = item['predicted_mos']
        utmos_dict[fid] = score
            
    return utmos_dict
    
def main():
    
    parser = argparse.ArgumentParser(description="Multi-Speaker TTS Inference")
    parser.add_argument('--meta_dir', type=str, required=True, help=f"Path to meta directory (Consists of lang_test.csv)")
    parser.add_argument('--orig_sample_dir', type=str, required=True, help="Path to original sample directory (Consists of wav files)")
    parser.add_argument('--sample_dir', type=str, required=True, help="Path to sample directory (Consists of generated_wav, attn)")
    parser.add_argument('--target_experiments', type=str, nargs='+', default=None, help="Target experiment(s) to process")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for metrics")
    parser.add_argument('--log_f0', action='store_true', help="Apply log F0 error")
    
    args = parser.parse_args()
    
    print(f"Loading meta data from {args.meta_dir}")
    full_meta = pd.concat([pd.read_csv(os.path.join(args.meta_dir, f), sep='|') for f in os.listdir(args.meta_dir) if f.endswith('test.csv')])
    full_meta['normalized text'] = full_meta['text'].apply(normalize_text)
    id_to_transcription = full_meta.set_index('id')['normalized text'].to_dict()
    print(f"Loaded {len(id_to_transcription)} transcriptions")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.target_experiments is None:
        print("No target experiment specified. Processing all experiments in the logs directory.")
        target_experiments = sorted(os.listdir(args.sample_dir))
    else:
        print(f"Processing target experiment(s): {args.target_experiments}")
        target_experiments = args.target_experiments
    
    all_files = sorted(os.listdir(args.orig_sample_dir))
    for exp in tqdm(target_experiments, total=len(target_experiments), desc="Processing experiments", leave=False):
        
        metrics_dict = {
            "sample name": [],
            "gt_utmos": [],
            "gen_utmos": [],
            "gt_text": [],
            "gt_pred_text": [],
            "gt_cer": [],
            "gen_pred_text": [],
            "gen_cer": [],
            "mcd": [],
            "f0": [],
        }
        
        lang = exp.split('-')[-1].lower()
        lcode = LANG_TO_OMNIASR[lang]
        ckpt_step = CKPT_STEP_PER_LANG[lang]
        
        sample_names = sorted(os.listdir(os.path.join(args.sample_dir, exp, f"G_{ckpt_step}", "generated_wav")))
        total_counts = len(sample_names)
        
        assert set(sample_names).issubset(set(all_files))
        
        gt_sample_paths = [os.path.join(args.orig_sample_dir, f) for f in sample_names if f.endswith('.wav')]
        generated_sample_paths = [os.path.join(args.sample_dir, exp, f"G_{ckpt_step}", "generated_wav", f) for f in sample_names]
        
        metrics_dict['sample name'].extend(sample_names)
        
        # 1. Calculate UTMOS
        # print(f"Loading and resampling GT samples from {args.orig_sample_dir}")
        # gt_samples = [load_and_resample(os.path.join(args.orig_sample_dir, f)) for f in sample_names]
        
        # MOS_MODEL.to('cuda')
        # gt_mos = []
        # for sample in tqdm(gt_samples, total=len(gt_samples), desc="Calculating UTMOS for GT samples"):
        #     score = MOS_MODEL.predict(data=sample.to('cuda'), sr=TARGET_RATE)
        #     gt_mos.append(score)
        # metrics_dict['gt_utmos'].extend(gt_mos)
            
        # print(f"Loading and resampling generated samples from {os.path.join(args.sample_dir, exp, f"G_{ckpt_step}", "generated_wav")}")
        # generated_samples = [load_and_resample(os.path.join(args.sample_dir, exp, f"G_{ckpt_step}", "generated_wav", f)) for f in sample_names]
        
        # gen_mos = []
        # for sample in tqdm(generated_samples, total=len(generated_samples), desc="Calculating UTMOS for generated samples"):
        #     score = MOS_MODEL.predict(data=sample.to('cuda'), sr=TARGET_RATE)
        #     gen_mos.append(score)
        # metrics_dict['gen_utmos'].extend(gen_mos)
        
        sample_ids = [f.split('.')[0] for f in sample_names]
        mos_gt = MOS_MODEL.predict(input_dir=args.orig_sample_dir,
                                   val_list=sample_ids,
                                   sr=22050)
        mos_gt = _unwrap_utmos(mos_gt)
        scores_gt = [mos_gt[fid] for fid in sample_ids]
        
        mos_gen = MOS_MODEL.predict(input_dir=os.path.join(args.sample_dir, exp, f"G_{ckpt_step}", "generated_wav"),
                                   val_list=sample_ids,
                                   sr=22050)
        mos_gen = _unwrap_utmos(mos_gen)
        scores_gen = [mos_gen[fid] for fid in sample_ids]
        
        metrics_dict['gt_utmos'].extend(scores_gt)
        metrics_dict['gen_utmos'].extend(scores_gen)
        
        # 2. Calculate CER
        gt_transcripts = [id_to_transcription[f.split('.')[0]] for f in sample_names]
        pred_transcripts_from_gt = ASR_MODEL.transcribe(gt_sample_paths, lang=[lcode for _ in sample_names], batch_size=32)
        pred_transcripts_from_generated = ASR_MODEL.transcribe(generated_sample_paths, lang=[lcode for _ in sample_names], batch_size=32)
        metrics_dict['gt_text'].extend(gt_transcripts)
        metrics_dict['gt_pred_text'].extend(pred_transcripts_from_gt)
        metrics_dict['gen_pred_text'].extend(pred_transcripts_from_generated)
        
        gt_cers = [cer(gt, pred) for gt, pred in zip(gt_transcripts, pred_transcripts_from_gt)]
        generated_cers = [cer(gt, pred) for gt, pred in zip(gt_transcripts, pred_transcripts_from_generated)]
        metrics_dict['gt_cer'].extend(gt_cers)
        metrics_dict['gen_cer'].extend(generated_cers)
        
        # Prepare for parallel computation for CPU-only systems
        pairs = list(zip(gt_sample_paths, generated_sample_paths))

        # 3. Calculate MCD (Order-preserving)
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            futures = [
                ex.submit(_mcd_worker, idx, p) 
                for idx, p in enumerate(pairs)
            ]
            
            mcds = [None] * len(pairs)

            for f in tqdm(as_completed(futures), total=len(futures), desc="Calculating MCD"):
                idx, result = f.result()
                mcds[idx] = result

        metrics_dict["mcd"].extend(mcds)


        # 4. Calculate F0 Error (Order-preserving)
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            futures = [
                ex.submit(_f0_worker, idx, p, args.log_f0) 
                for idx, p in enumerate(pairs)
            ]
            
            f0s = [None] * len(pairs)

            for f in tqdm(as_completed(futures), total=len(futures), desc="Calculating F0 Error"):
                idx, result = f.result()
                f0s[idx] = result

        metrics_dict["f0"].extend(f0s)

        
        metrics_df = pd.DataFrame(metrics_dict)
        metrics_df.to_csv(os.path.join(args.output_dir, f"{exp}.csv"), index=False)
            
if __name__ == "__main__":
    main()
    
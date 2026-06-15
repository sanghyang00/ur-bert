import argparse
import os
import json
import phonemizer
import pandas as pd
import torch
from tqdm import tqdm

import os, re, json, argparse
import pandas as pd
from tqdm import tqdm

import uroman as ur
import pykakasi
from pypinyin import lazy_pinyin
from pycantonese import characters_to_jyutping

ROMANIZER = ur.Uroman() 
KKS = pykakasi.kakasi() 

LANG_TO_ISO = {
    'en': 'eng',
    'zh': 'cmn',
    'de': 'deu',
    'jv': 'jav',
    'su': 'sun',
    'km': 'khm',
    'np': 'nep',
    'af': 'afr',
    'st': 'sot',
    'tn': 'tsn',
    'xh': 'xho',
    'si': 'sin',
}

ABB_MAPPING_LJ = {
    "Mr.": "Mister",
    "Mrs.": "Misess",
    "Dr.": "Doctor",
    "No.": "Number",
    "St.": "Saint",
    "Co.": "Company",
    "Jr.": "Junior",
    "Maj.": "Major",
    "Gen.": "General",
    "Drs.": "Doctors",
    "Rev.": "Reverend",
    "Lt.": "Lieutenant",
    "Hon.": "Honorable",
    "Sgt.": "Sergeant",
    "Capt.": "Captain",
    "Esq.": "Esquire",
    "Ltd.": "Limited",
    "Col.": "Colonel",
    "Ft.": "Fort",
}

PUNCT_MAPPING_LJ = {
    '.': ' ',
    '-': ' ',
}

def romanize_string(transcription, iso_code, apply_pinyin=True):
    if iso_code == 'jpn':
        romanized_transcription = ''.join([i['hepburn'] for i in KKS.convert(transcription)])
    elif iso_code == 'cmn' and apply_pinyin:
        romanized_transcription = ''.join(lazy_pinyin(transcription)).strip()
    elif iso_code == 'yue':
        romanized_transcription = ''.join([i[-1] for i in characters_to_jyutping(transcription) if i[-1] is not None])
        romanized_transcription = re.sub(r'\d', '', romanized_transcription)
    else:
        romanized_transcription = ROMANIZER.romanize_string(transcription, lcode=iso_code)
    
    romanized_transcription = romanized_transcription.lower()
    romanized_transcription = re.sub(r'[^a-zA-Z\s]', '', romanized_transcription)
    romanized_transcription = re.sub(r'\s+', ' ', romanized_transcription).strip()
    return romanized_transcription

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_dir", default="/ssd2/sangmin/datasets/tts_db/tts_metadata")
    parser.add_argument("--speaker_dir", default="/ssd2/sangmin/datasets/tts_db/speaker_mappings")
    parser.add_argument("--wav_dir", default="/ssd2/sangmin/datasets/tts_db/trimmed_wavs")
    parser.add_argument("--output_dir", default="data/urbert")
    parser.add_argument("--langs", default=None, type=str, nargs='+')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    filelists = [f for f in os.listdir(args.metadata_dir) if f.endswith('.csv')]
    langs = sorted(set([file.split('_')[0] for file in filelists]))
    
    supported_langs = [lang for lang in langs if LANG_TO_ISO.get(lang, '') != '']

    if args.langs is not None:
        supported_langs = args.langs

    for lang in supported_langs:
        print(f"\n--- Processing: {lang} ---")
        
        json_path = os.path.join(args.speaker_dir, f"{lang}_speakers.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_map = json.load(f)
            
            spk_map = {}
            for k, v in raw_map.items():
                k_str = str(k).strip()
                if lang in ['jv', 'su']:
                  new_key = f"{int(k_str):05d}" if k_str.isdigit() else k_str
                else:
                  new_key = f"{int(k_str):04d}" if k_str.isdigit() else k_str
                spk_map[new_key] = str(v).strip()

        train = pd.read_csv(os.path.join(args.metadata_dir, f"{lang}_train.csv"), 
                            sep='|', dtype=str, keep_default_na=False)
        test = pd.read_csv(os.path.join(args.metadata_dir, f"{lang}_test.csv"), 
                           sep='|', dtype=str, keep_default_na=False)

        if lang == 'en':
            print(f'Processing {lang} abbreviations...')
            for k, v in ABB_MAPPING_LJ.items():
                train['text'] = train['text'].str.replace(k, v, regex=False)
                test['text'] = test['text'].str.replace(k, v, regex=False)
            
            for k, v in PUNCT_MAPPING_LJ.items():
                train['text'] = train['text'].str.replace(k, v, regex=False)
                test['text'] = test['text'].str.replace(k, v, regex=False)
        
        for df, mode in zip([train, test], ['train', 'test']):
            print(f"Romanizing {mode} data (Batch Processing)...")
            texts = df['text'].astype(str).tolist()
            phonemes = []
            for text in tqdm(texts, desc=f"Romanizing {lang} {mode} data"):
              phonemes.append(romanize_string(text, LANG_TO_ISO[lang]))
            df['phoneme_seq'] = phonemes
            
            out = df[['id', 'speaker', 'phoneme_seq']].copy()
            out['id'] = out['id'].apply(lambda x: f"{args.wav_dir}/{x}.wav")
            out['speaker'] = out['speaker'].str.strip()
            out['speaker_mapped'] = out['speaker'].map(spk_map)
            
            if out['speaker_mapped'].isnull().any():
                fail_ids = out[out['speaker_mapped'].isnull()]['speaker'].unique()
                print(f"!!! [Warning] {lang} {mode} mapping failed for speaker IDs (dropped): {fail_ids}")
            
            out['speaker'] = pd.to_numeric(out['speaker_mapped']).astype(int)
            
            final_out = out[['id', 'speaker', 'phoneme_seq']]
            
            save_path = os.path.join(args.output_dir, f"{lang}_{mode}.txt")
            final_out.to_csv(save_path, sep='|', index=False, header=False)
            print(f"Successfully saved {len(final_out)} rows to {save_path}")

if __name__ == '__main__':
    main()
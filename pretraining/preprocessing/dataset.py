import os, torch, torchaudio
import pandas as pd
from torch.utils.data import Dataset

from utils import normalize_speech

class AlignmentDataset(Dataset):
    def __init__(self, folder_paths, split='train', duration_limit=30.0):
        
        assert (split in ['train', 'dev', 'test'])

        # columns: file path, language, iso 639-3, iso 15924, transcription, romanized transcription, duration

        all_dfs = []
        print(f"Loading CSVs for split '{split}'...")
        if not isinstance(folder_paths, list):
            folder_paths = [folder_paths]
            
        for path in folder_paths:
            csv_path = os.path.join(path, f'{split}.csv')
            try:
                df = pd.read_csv(csv_path)
                corpus_name = os.path.basename(path.rstrip('/'))
                df['corpus'] = corpus_name
                all_dfs.append(df)
                print(f"  Loaded {len(df)} rows from {csv_path}")
            except FileNotFoundError:
                print(f"  Warning: CSV not found, skipping: {csv_path}")
        
        if not all_dfs:
            raise FileNotFoundError(f"No CSV data found for split '{split}' in paths: {folder_paths}")
            
        self.data = pd.concat(all_dfs, ignore_index=True)
        self.data = self.data.dropna(subset=['romanized transcription']).reset_index(drop=True)
        self.data = self.data[self.data['duration'] >= 1.0].reset_index(drop=True) # minimum duration of 1.0s
        self.data = self.data[self.data['duration'] <= duration_limit].reset_index(drop=True)

        self.duration = round(self.data['duration'].sum() / 3600, 2)
        self.lengths = self.data['duration'].tolist()
        
        print("Initialization complete!")

        print('--------------------------------')
        print(f"Total rows loaded: {len(self.data)}")
        print(f"Total duration: {self.duration} hours")
        print(f"Duration range: {self.data['duration'].min():.2f}s ~ {self.data['duration'].max():.2f}s")
        print(f"Number of languages: {len(self.data['language'].unique())}")
        print('--------------------------------')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        max_retries = 5
        original_idx = idx
        attempts = 0
        current_idx = idx

        while attempts < max_retries:
            try:
                row = self.data.iloc[current_idx]
                audio_path = row['file path']
                roman = row['romanized transcription'].strip()

                audio, sr = torchaudio.load(audio_path)
                if sr != 16000:
                    audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
                audio = normalize_speech(audio)
                audio = audio.squeeze()

                if torch.isnan(audio).any():
                    raise ValueError(f"NaN found in audio: {audio_path}")

                return audio_path, audio, roman

            except Exception as e:
                attempts += 1
                print(f"[Attempt {attempts}/{max_retries}] Failed to load {audio_path}: {str(e)}")
                
                current_idx = (current_idx + 1) % len(self.data)

        last_attempted_path = self.data.iloc[current_idx]['file path']
        raise RuntimeError(
            f"Failed to load a valid audio sample after {max_retries} attempts.\n"
            f"Original index: {original_idx}\n"
            f"Last attempted file: {last_attempted_path}\n"
            f"Check if the file exists, is readable, and not corrupted."
        )

class KMeansDataset(Dataset):
    def __init__(self, folder_paths, split='train',
                 beta_l=0.5, beta_d=0.5, duration_limit=30.0, max_samples_per_lang=100):
        
        assert (split in ['train', 'dev', 'test'])

        # columns: file path, language, iso 639-3, iso 15924, transcription, romanized transcription, duration

        all_dfs = []
        print(f"Loading CSVs for split '{split}'...")
        if not isinstance(folder_paths, list):
            folder_paths = [folder_paths]
            
        for path in folder_paths:
            csv_path = os.path.join(path, f'{split}.csv')
            try:
                df = pd.read_csv(csv_path)
                corpus_name = os.path.basename(path.rstrip('/'))
                df['corpus'] = corpus_name
                all_dfs.append(df)
                print(f"  Loaded {len(df)} rows from {csv_path}")
            except FileNotFoundError:
                print(f"  Warning: CSV not found, skipping: {csv_path}")
        
        if not all_dfs:
            raise FileNotFoundError(f"No CSV data found for split '{split}' in paths: {folder_paths}")
            
        self.data = pd.concat(all_dfs, ignore_index=True)
        self.data = self.data.dropna(subset=['romanized transcription']).reset_index(drop=True)
        self.data = self.data[self.data['duration'] >= 1.0].reset_index(drop=True) # minimum duration of 1.0s
        self.data = self.data[self.data['duration'] <= duration_limit].reset_index(drop=True)

        self.duration = round(self.data['duration'].sum() / 3600, 2)
        self.lengths = self.data['duration'].tolist()
        
        # if split == 'train':
        #     print(f"Calculating Two-Stage Weights (Beta_L={beta_l}, Beta_D={beta_d})...")
            
        #     self.data['count_LC'] = self.data.groupby(['lcode', 'corpus'])['file path'].transform('count')
            
        #     self.data['count_L'] = self.data.groupby('lcode')['file path'].transform('count')
            
        #     self.data['score_C_given_L'] = self.data['count_LC'].pow(beta_l)
            
        #     sum_score_by_L = self.data.groupby('lcode')['score_C_given_L'].transform('sum')
        #     self.data['prob_C_given_L'] = self.data['score_C_given_L'] / sum_score_by_L
            
        #     total_count = len(self.data)
        #     self.data['score_L'] = (self.data['count_L'] / total_count).pow(beta_d)
            
        #     sum_score_L = self.data.groupby('lcode')['score_L'].first().sum()
        #     self.data['prob_L'] = self.data['score_L'] / sum_score_L
            
        #     weight_factors = (self.data['score_L'] * self.data['prob_C_given_L']) / self.data['count_LC']
            
        #     self.samples_weights = torch.tensor(weight_factors.values, dtype=torch.double)
            
        #     drop_cols = ['count_LC', 'count_L', 'score_C_given_L', 'prob_C_given_L', 'score_L', 'prob_L']
            
        #     cols_to_drop = [c for c in drop_cols if c in self.data.columns]
        #     self.data.drop(columns=cols_to_drop, inplace=True)
            
        print(f"Sampling max {max_samples_per_lang} rows per language for split '{split}'...")
        
        self.data = (
            self.data.sample(frac=1, random_state=42)
            .groupby('lcode')
            .head(max_samples_per_lang)
            .reset_index(drop=True)
        )
        
        print(f"  -> Reduced total rows to: {len(self.data)}")
        
        print("Initialization complete!")

        print('--------------------------------')
        print(f"Total rows loaded: {len(self.data)}")
        print(f"Total duration: {self.duration} hours")
        print(f"Duration range: {self.data['duration'].min():.2f}s ~ {self.data['duration'].max():.2f}s")
        print(f"Number of languages: {len(self.data['language'].unique())}")
        print('--------------------------------')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        while True:
            try:
                row = self.data.iloc[idx]
                audio_path = row['file path']

                audio, sr = torchaudio.load(audio_path)
                if sr != 16000:
                    audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
                    sr = 16000
                assert (sr == 16000)
                audio = normalize_speech(audio)
                
                audio = audio.squeeze()
                
                if torch.isnan(audio).any():
                    raise ValueError(f"NaN found in audio at file: {audio_path}")
                    
                return audio
            
            except Exception as e:
                print(e)
                idx = (idx + 1) % len(self.data)
                continue

# URBERT Training

This repository contains training code and configs for URBERT language-model experiments, including:

- MLM-only training
- MLM + distillation training

## Project Structure

```text
urbert_training/
├── configs/                      # Experiment config files
│   ├── urbert_MLM.yaml
│   └── urbert_MLM_distill.yaml
├── csvs/                         # Source CSVs (CommonVoice, FLEURS, OAC)
├── data/                         # Processed CSVs used for training
├── logs/                         # Training logs and experiment outputs
├── dataloader.py                 # URbertDataset + custom collator
├── model.py                      # MultiTaskBert model
├── tokenizer.py                  # UromanCharTokenizer
├── train.py                      # Main training entry point
├── requirements.txt              # Python dependencies
└── README.md
```

## Data Preparation Note (dev split update)

To add/update development data with token mapping:

- Refer to: `/workspace/{username}/urbert/audio_tokens/data.ipynb`
- Update the `token_mapping.txt` path to your current environment.
- Merge mapped fields into the existing CSVs.

Expected output naming:

- Original files: `train_original.csv`, `dev_original.csv`
- Updated files: `train.csv`, `dev.csv`

The notebook is already structured so you mainly need to fix paths.  
Typical runtime is around 5 minutes.

## Run Training

Set project directory first:

```bash
cd /workspace/AEG_intern/urbert
```

Set the GPU index with `CUDA_VISIBLE_DEVICES` before running.

### 1) MLM only

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/urbert_MLM.yaml \
  --exp_name urbert_MLM \
  --resume None
```

Background run:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
  --config configs/urbert_MLM.yaml \
  --exp_name urbert_MLM \
  --resume None \
  > logs/nohup/urbert_MLM.log 2>&1 &
```

### 2) MLM + Distillation

```bash
CUDA_VISIBLE_DEVICES=3 python train.py \
  --config configs/urbert_MLM_distill.yaml \
  --exp_name urbert_MLM_distill \
  --resume None
```

Background run:

```bash
CUDA_VISIBLE_DEVICES=3 nohup python train.py \
  --config configs/urbert_MLM_distill.yaml \
  --exp_name urbert_MLM_distill \
  --resume None \
  > logs/nohup/urbert_MLM_distill.log 2>&1 &
```

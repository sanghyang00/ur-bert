# URBERT Finetuning

This repository contains finetuning code and configs for VITS-based multilingual TTS experiments with pretrained text encoders, including:

- PLBERT
- XPhoneBERT
- UR-BERT
- VITS baseline (no pretrained BERT encoder)

## Project Structure

```text
finetuning/
├── configs/                          # Experiment configs by encoder/language
│   ├── plbert/
│   ├── xphonebert/
│   ├── urbert/
│   ├── urbert_abl/
│   └── vits/
├── data/                             # Formatted filelists for training/inference
│   └── urbert/
├── logs/                             # Training outputs (created at runtime)
├── preprocess_for_plbert.py          # Build filelists for PLBERT setup
├── preprocess_for_xphonebert.py      # Build filelists for XPhoneBERT setup
├── preprocess_for_urbert.py          # Build filelists for UR-BERT setup
├── train.py                          # Main training entry point
├── inference.py                      # Batch inference script for checkpoints
├── load_encoder.py                   # Encoder/tokenizer loader by bert_type
├── models.py                         # VITS generator + discriminator modules
├── data_utils.py                     # Dataset, collate, sampler utilities
├── requirements.txt                  # Python dependencies
└── README.md
```

## Data Preparation

Prepare language-specific train/test filelists in `id|speaker|text_or_phoneme` format:

- PLBERT pipeline: `preprocess_for_plbert.py`
- XPhoneBERT pipeline: `preprocess_for_xphonebert.py`
- UR-BERT pipeline: `preprocess_for_urbert.py`

Generated files are saved under `data/<encoder_type>/`, for example:

- `data/urbert/en_train.txt`
- `data/urbert/en_test.txt`

## Run Finetuning

Set project directory first:

```bash
cd /ssd/sangmin/urbert/finetuning
```

Set GPU index with `CUDA_VISIBLE_DEVICES` before running.

### 1) UR-BERT finetuning

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --bert_type urbert \
  --config configs/urbert/english.json \
  --experiment_name urbert-english
```

### 2) PLBERT finetuning

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --bert_type plbert \
  --config configs/plbert/english.json \
  --experiment_name plbert-english
```

### 3) XPhoneBERT finetuning

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --bert_type xphonebert \
  --config configs/xphonebert/english.json \
  --experiment_name xphonebert-english
```

### 4) VITS baseline finetuning

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --bert_type vits \
  --config configs/vits/english.json \
  --experiment_name vits-english
```

## Run Inference

Run inference from saved experiment directories in `logs/`:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --logs_dir logs \
  --output_dir outputs \
  --target_experiments urbert-english \
  --device cuda
```

Use `--predefined_training_step` if you want fixed checkpoint steps by language.

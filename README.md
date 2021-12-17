<div align="center">

<img src="docs/source/imgs/logo.png" width="400px">

**Easy Language Model Pretraining leveraging Huggingface's [Transformers](https://github.com/huggingface/transformers) and [Datasets](https://github.com/huggingface/datasets)**

<p align="center">
  <a href="#what-is-lassl">What is LASSL</a> •
  <a href="#how-to-use">How to Use</a>
</p>

<p>
    <b>English</b> |
    <a href="README_ko.md">한국어</a>
</p>

<p align="center">
    <a href="https://github.com/lassl/lassl/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg">
    </a>
    <a href="https://github.com/lassl/lassl/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/lassl/lassl">
    </a>
    <a href="https://huggingface.co/lassl">
        <img alt="Issues" src="https://img.shields.io/badge/huggingface-lassl-002060">
    </a>
</p>

</div>

## What is LASSL
LASSL is a **LA**nguage library for **S**elf-**S**upervised **L**earning. LASSL aims to provide a easy-to-use framework for pretraining language model by only using Huggingface's Transformers and Datasets.

## Environment setting
You can install the required packages following:
```bash
pip3 install -r requirements.txt
```

or you can set environment with poetry following:
```bash
# Install poetry 
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
# Environment setting with poetry
poetry install
```


## How to Use
- Language model pretraining can be divided into three steps: **1. Train Tokenizer**, **2. Serialize Corpus**, **3.Pretrain Language Model**.
- After preparing corpus following to <a href="https://github.com/lassl/lassl/blob/main/docs/supported_dataset.md">supported corpus type</a>, you can pretrain your own language model.

### 1. Train Tokenizer
```bash
python3 train_tokenizer.py \
    --corpora_dir $CORPORA_DIR \
    --corpus_type $CORPUS_TYPE \
    --sampling_ratio $SAMPLING_RATIO \
    --model_type $MODEL_TYPE \
    --vocab_size $VOCAB_SIZE \
    --min_frequency $MIN_FREQUENCY
```

```bash
# poetry 이용
poetry run python3 train_tokenizer.py \
    --corpora_dir $CORPORA_DIR \
    --corpus_type $CORPUS_TYPE \
    --sampling_ratio $SAMPLING_RATIO \
    --model_type $MODEL_TYPE \
    --vocab_size $VOCAB_SIZE \
    --min_frequency $MIN_FREQUENCY
```

### 2. Serialize Corpora
```bash
python3 serialize_corpora.py \
    --model_type $MODEL_TYPE \
    --tokenizer_dir $TOKENIZER_DIR \
    --corpora_dir $CORPORA_DIR \
    --corpus_type $CORPUS_TYPE \
    --max_length $MAX_LENGTH \
    --num_proc $NUM_PROC
```

```bash
# with poetry
poetry run python3 serialize_corpora.py \
    --model_type $MODEL_TYPE \
    --tokenizer_dir $TOKENIZER_DIR \
    --corpora_dir $CORPORA_DIR \
    --corpus_type $CORPUS_TYPE \
    --max_length $MAX_LENGTH \
    --num_proc $NUM_PROC
```

### 3. Pretrain Language Model
```bash
python3 pretrain_language_model.py --config_path $CONFIG_PATH
```

```bash
# with poetry
poetry run python3 pretrain_language_model.py --config_path $CONFIG_PATH
```

```bash
# When using TPU, use the command below. (Poetry environment does not provide PyTorch XLA as default.)
python3 xla_spawn.py --num_cores $NUM_CORES pretrain_language_model.py --config_path $CONFIG_PATH
```

## Contributors
Boseop Kim|Minho Ryu|Inje Ryu|Jangwon Park|Hyoungseok Kim
:-:|:-:|:-:|:-:|:-:
![image1][image1]|![image2][image2]|![image3][image3]|![image4][image4]|![image5][image5]
[Github](https://github.com/seopbo)|[Github](https://github.com/bzantium)|[Github](https://github.com/iron-ij)|[Github](https://github.com/monologg)|[Github](https://github.com/alxiom)

[image1]: https://avatars.githubusercontent.com/seopbo
[image2]: https://avatars.githubusercontent.com/bzantium
[image3]: https://avatars.githubusercontent.com/iron-ij
[image4]: https://avatars.githubusercontent.com/monologg
[image5]: https://avatars.githubusercontent.com/alxiom

## Acknowledgements
LASSL is built with Cloud TPU support from the Tensorflow Research Cloud (TFRC) program.

# NarrativeBridge: Enhancing Video Captioning with Causal-Temporal Narrative

**ICLR 2025** | [Paper](https://openreview.net/forum?id=bBoetBIN2R) | [arXiv](https://arxiv.org/abs/2406.06499) | [Project Page](https://narrativebridge.github.io) | [Datasets on HuggingFace](https://huggingface.co/narrativebridge)

> **Authors:** Asmar Nadeem, Faegheh Sardari, Robert Dawes, Syed Sameed Husain, Adrian Hilton, Armin Mustafa
> **Affiliation:** University of Surrey, UK

---

## Overview

NarrativeBridge addresses a fundamental limitation of existing video captioning models: they describe *what* happens in a video but not *why* — the causal-temporal structure linking events. We introduce:

1. **Causal-Temporal Narrative (CTN) Benchmark** — two new datasets (MSRVTT-CTN and MSVD-CTN) with captions that explicitly encode cause-and-effect temporal relationships, generated via LLM few-shot prompting and validated by human annotators.

2. **Cause-Effect Network (CEN)** — a two-stream architecture with **separate visual encoders** for the cause and effect dynamics of a video. Their outputs are concatenated and decoded by a BERT-initialised transformer to produce causally-grounded captions.

**Example:** instead of *"a car flips over"*, CEN generates *"a car loses control after speeding and flips over, causing the driver to be ejected"*.

---

## Method

CEN is a two-stage pipeline using only standard transformer modules.

### Stage 1 — CLIP4Clip cause / effect feature extraction

Train two separate [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) models — one supervised on the **cause** captions of the CTN datasets, one on the **effect** captions. Use each to extract per-video patch-token features:

```
Video → CLIP4Clip (cause-trained)  → cause_features.pickle
Video → CLIP4Clip (effect-trained) → effect_features.pickle
```

Scripts: `stage1_feature_extraction/extract_features_cause.py` and `extract_features_effect.py`.

### Stage 2 — Cause-Effect Network (CEN)

The cause and effect feature streams are each passed through a small transformer visual encoder (`VisualModel`, configured with `visual_num_hidden_layers=2`). Their per-frame token sequences are then concatenated along the sequence dimension (with concatenated attention masks) and decoded by a BERT-initialised transformer decoder.

```
cause_features  → VisualEncoder_cause  ──┐
                                          ├─ concat → BERT Decoder → Caption
effect_features → VisualEncoder_effect ──┘
```

Core module: `modules/modeling_ce.py` — `CaptionGenerator`.

---

## Datasets

The CTN datasets are available on HuggingFace (CC BY-NC-ND 4.0):

```bash
# MSVD-CTN
huggingface-cli download asmarnadeem/NarrativeBridge --repo-type dataset

# Or via Python
from datasets import load_dataset
dataset = load_dataset("asmarnadeem/NarrativeBridge")
```

For MSR-VTT, download the raw videos from the [official MSR-VTT page](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/) and use the CTN annotation JSON.

---

## Installation

```bash
conda env create -f environment.yml
conda activate narrativebridge
```

Or with pip:

```bash
pip install -r requirements.txt
```

**Key dependencies:** PyTorch 2.1, HuggingFace Transformers 4.36, pycocoevalcap.

---

## Usage

### Step 1 — Train CLIP4Clip cause and effect models

Follow the [CLIP4Clip repository](https://github.com/ArrowLuo/CLIP4Clip) to train two models:
- one using the **cause** captions from MSRVTT-CTN / MSVD-CTN
- one using the **effect** captions

### Step 2 — Extract cause and effect features

Edit the `CAUSE_MODEL_PATH` / `EFFECT_MODEL_PATH` variables in each script, then run:

```bash
cd stage1_feature_extraction

# MSR-VTT
python extract_features_cause.py
python extract_features_effect.py

# MSVD (set args.msvd = True in the script)
python extract_features_cause.py
python extract_features_effect.py
```

Output: `extracted/msrvtt/MSRVTT_cause_features.pickle` and `MSRVTT_effect_features.pickle`.

### Step 3 — Train CEN

Edit the path variables in the shell script, then:

```bash
# MSR-VTT
bash scripts/train_msrvtt.sh

# MSVD (fine-tune from the MSR-VTT checkpoint)
bash scripts/train_msvd.sh
```

### Step 4 — Evaluate

```bash
bash scripts/eval_msrvtt.sh
bash scripts/eval_msvd.sh
```

---

## Repository Structure

```
NarrativeBridge/
├── stage1_feature_extraction/    # Stage 1: CLIP4Clip-based cause/effect feature extraction
│   ├── extract_features_cause.py
│   ├── extract_features_effect.py
│   ├── modules/                  # CLIP4Clip model modules
│   └── dataloaders/
├── modules/                      # Stage 2: CEN model modules
│   ├── modeling_ce.py            # ★ CaptionGenerator: dual visual encoder + decoder
│   ├── module_visual.py          # Transformer visual encoder (used for both streams)
│   ├── module_decoder.py         # BERT-initialised caption decoder
│   ├── module_bert.py            # BERT modules for decoder weight init
│   ├── module_clip.py            # CLIP layers (used by stage 1)
│   ├── beam.py                   # Beam search at inference
│   ├── optimization.py           # BertAdam optimiser
│   ├── tokenization.py           # BERT tokenizer
│   ├── until_module.py, until_config.py, file_utils.py
│   ├── bert-model/, decoder-base/, visual-base/    # Pretrained configs
├── dataloaders/                  # Stage 2 dataloaders for MSRVTT-CTN and MSVD-CTN
│   ├── dataloader_msrvtt_caption_ce.py
│   └── dataloader_msvd_caption_ce.py
├── engine.py                     # ★ Shared train/eval logic, beam search, metrics
├── args.py                       # Shared argparse builder for the 4 entry scripts
├── utils.py                      # get_logger
├── train_msrvtt.py, train_msvd.py
├── eval_msrvtt.py,  eval_msvd.py
├── metrics.py                    # BLEU / ROUGE-L / CIDEr evaluation utilities
└── scripts/                      # Training and evaluation shell scripts
```

---

## Key Hyperparameters

| Hyperparameter         | MSRVTT-CTN | MSVD-CTN     |
|------------------------|------------|--------------|
| Learning rate          | 5e-5       | 5e-7         |
| Batch size             | 64         | 64           |
| Max frames             | 20         | 20           |
| Max words              | 48         | 48           |
| Visual hidden layers   | 2          | 2            |
| Decoder hidden layers  | 2          | 2            |
| Video feature dim      | 512        | 512          |
| d_model                | 512        | 512          |
| Beam size (inference)  | 5          | 5            |

---

## Citation

If you use NarrativeBridge, the CTN datasets, or this code in your research, please cite:

```bibtex
@inproceedings{nadeem2025narrativebridge,
  title     = {NarrativeBridge: Enhancing Video Captioning with Causal-Temporal Narrative},
  author    = {Nadeem, Asmar and Sardari, Faegheh and Dawes, Robert and Husain, Syed Sameed and Hilton, Adrian and Mustafa, Armin},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=bBoetBIN2R}
}
```

---

## License

The code in this repository is released under the MIT License.
The MSRVTT-CTN and MSVD-CTN datasets are released under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

---

## Acknowledgements

The CEN decoder builds on [UniVL](https://github.com/microsoft/UniVL). Stage 1 feature extraction uses [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip).

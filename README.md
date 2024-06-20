# Avengers!

This repo serves as a workshop repo to experiment on Transformers-alternatives that 
use constant cache size and thus improve inference efficiencies.

## Teaming
We are developing this in the open using data available in the community with the primary data sources from [Dolma](https://allenai.github.io/dolma/) and [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). The checkpoints will be made available to the community as well for further studies. So far, we have been in initial discussions with Minjia Zhang and Charith Mendis from UIUC, Tri Dao from Princeton, and Albert Gu from CMU. Anyone interested, please reach out to Raghu Ganti (rganti@us.ibm.com).

## Roadmap
We put together a brief [roadmap](Roadmap.md) on studying these architectures and welcome input on suggestions, which we will take into account when we run the jobs.

## Models

- Llama-3 (Transformers baseline)
- [Mamba-2](https://arxiv.org/pdf/2312.00752) (with support for hybrid Mamba - mix of Mamba layers and Attn layers)
- [Telescoping Cache Llama-3](docs/telescoping-cache.md) (constant-cache-size implementation of Transformers)

## Installation

We leverage the following repos for our experiments:

### Models
[Mamba](https://github.com/state-spaces/mamba): provides the model implementation for Mamba-2:
```bash
  git clone https://github.com/Dao-AILab/causal-conv1d.git
  cd causal-conv1d && pip install -e . && cd ..
  git clone https://github.com/state-spaces/mamba.git
  cd mamba && pip install -e . && cd ..
  pip install flash-attn --no-build-isolation
```
[Foundation-Model-Stack](https://github.com/foundation-model-stack/foundation-model-stack): 
provides the model implementation for Llama-3.
```bash
git clone https://github.com/foundation-model-stack/foundation-model-stack.git
cd foundation-model-stack && pip install -e . && cd ..
```
[Foundation-Model-Stack-Sandbox](https://github.com/daviswer/foundation-model-stack-sandbox):
provides the model implementation for Telescoping Cache Llama-3. (temporary location,
to be moved to main Foundation-Model-Stack)
```bash
git clone -b telescoping-cache https://github.com/daviswer/foundation-model-stack-sandbox.git
cd foundation-model-stack-sandbox && pip install -e . && cd ..
```

### Training Stack
[fms-fsdp](https://github.com/foundation-model-stack/fms-fsdp): provides the training stack
for model training


## Run

modify [train_llama.sh](train_llama.sh) / [train_mamba.sh](train_mamba.sh) / 
[train_tele.sh](train_tele.sh) to run the training.


## Models available -- COMING SOON!

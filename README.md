# Avengers!

This repo serves as a workshop repo to experiment on Transformers-alternatives that 
use constant cache size.

## Models

- Llama-3 (Transformers baseline)
- Mamba-2 (with support for hybrid Mamba - mix of Mamba layers and Attn layers)
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

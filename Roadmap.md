The current plan is that, for each model variant, the followings will be run in order:

1. 3b run with 300b tokens (cc only, using [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu))
2. 3b run with 1T tokens (mixed data, using [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) + [Dolma v1.7](https://huggingface.co/datasets/allenai/dolma))
3. 7b run with 2T-4T tokens (TBD)

models will be [evaluated](https://github.com/EleutherAI/lm-evaluation-harness) and reported back in this repo.

We will keep a running log of the experiments and rough schedules (depends on GPU availability).

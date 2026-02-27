# AlphaGenome_PyTorch
Repository for AlphaGenome_PyTorch.



## Docs
The `docs` directory contains instructions on environment setup, explanations of the data structure and model architecture, and running examples. It's strongly recommended to read the `model.md` and `data.md` markdown files in the `*/AlphaGenome_PyTorch/docs/guides` directory before running examples so that you can understand why the metadata and dummy data are structured the way they are.

### Environment
See `*/AlphaGenome_PyTorch/docs/environment` for instructions on how to set up a UV environment to run AlphaGenome_PyTorch.

### Guides
See `*/AlphaGenome_PyTorch/docs/guides` for explanations on the AlphaGenome model and its data structure (very helpful for understanding examples).

### Examples
See `*/AlphaGenome_PyTorch/docs/examples` for examples of:
- Masked Language Modeling (MLM) training (`train_mlm.py`)
- Training on Downstream Tasks (RNA-Seq, CAGE, ATAC, Splice Sites Classification/Usage/Junction) (`train_downstream.py`)
- MLM Pretraining --> Training on Downstream Tasks (`train_downstream_from_pretrained.py`)


## Licensing
The repository is a reimplementation of the AlphaGenome model in PyTorch, along with the added option to do Masked Language Modeling. Within the `alphagenome_pytorch` directory, some components are direct ports of the released AlphaGenome code, some are reimplementations from the paper's (fairly verbose) pseudocode, and others are added by me (e.g. MLM head) - the attribution is made clear at the top of each file in the `alphagenome_pytorch` directory. Aside from that, the `docs` and `tests` directories are written by me (with LLM coding assistance).


## Licensing
This repository is a reimplementation of the AlphaGenome model in PyTorch, with an added option for Masked Language Modeling (MLM).

Within the `alphagenome_pytorch` directory, some components are direct ports of the released AlphaGenome code [Link1](https://github.com/google-deepmind/alphagenome) [Link2](https://github.com/google-deepmind/alphagenome_research) (licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)), some are reimplementations based on pseudocode from the [BioArXiV paper](https://www.biorxiv.org/content/10.1101/2025.06.25.661532v1), and others are original additions (e.g., the MLM head). Attribution is made clear at the top of each file in the `alphagenome_pytorch` directory. The `docs` and `tests` directories are original work (with LLM coding assistance).


## Intended Audience
This intended audience of this repo is model trainers: those who might want to take the AlphaGenome architecture and train it in a way that gives them some flexibility over hyperparameters, and/or do the training in PyTorch rather than Jax. If you can prepare a batch of tensor data and set up a train/val/test loop, but don't want the hassle of making sure every linear layer and norm is placed correctly while replicating the architecture, then this repo is for you. The added MLM pretraining head is also a plus.


## Reaching Out
Want a new feature or find a bug? Feel free to leave an issue on the GitHub repository.

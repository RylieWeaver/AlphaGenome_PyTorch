# Examples


## Setup
From the repo root, install the local package in editable mode, then enter the
examples directory:

```bash
cd */AlphaGenome_PyTorch
pip install -e .
cd docs/examples
```


## Running
Feel free to open the scripts and start changing things like optimizer, model hyperparameters, learning rate, etc... 
It's meant to be barebones and hackable so that you can start modifying to your needs!

```
python train_mlm.py
python train_downstream.py
python train_downstream_from_pretrained_mlm.py
python train_downstream_from_hf_checkpoint.py
```

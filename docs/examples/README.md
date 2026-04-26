# Examples


## Setup
(1) Have the necessary dependencies installed (see `*/AlphaGenome_PyTorch/docs/environment`) for details on setting up an environment.
(2) Make sure AlphaGenome_PyTorch is in your Python path: `cd ../.. && export PYTHONPATH=$(pwd):$PYTHONPATH && cd docs/examples`


## Running
Feel free to open the scripts and start changing things like optimizer, model hyperparameters, learning rate, etc... 
It's meant to be barebones and hackable so that you can start modifying to your needs!

```
python train_mlm.py
python train_downstream.py
python train_downstream_from_pretrained_mlm.py
python train_downstream_from_hf_checkpoint.py
```


## Optionally run tests
```
uv pip install pytest
cd `*/AlphaGenome_PyTorch`
pytest -q tests
```

# SciPy 2026 measurements

With the project environment active, enter this directory before running the
suite. GPU benchmarks and training runs require CUDA. All raw results and 
generated figures are written under `scipy-proceedings/results`.

```bash
cd scipy-proceedings
```

## Parameter counts

```bash
python parameter_counts.py
```

## Performance (Inference/Training)

```bash
python performance/inference.py --mode predictions
python performance/inference.py --mode embeddings
python performance/training.py --mode predictions
python performance/training.py --mode embeddings
```

Plot both performance experiments after their four runs finish:

```bash
python performance/plot.py
```

## Training curves

### Data

```bash
python -m pip install pysam pyBigWig
python training_curves/download_data.py
```
### Training

Run both experiments from scratch and from checkpoint initialization:

```bash
python training_curves/train_mlm.py
python training_curves/train_mlm.py --load-state
python training_curves/train_rna_seq.py
python training_curves/train_rna_seq.py --load-state
```

Model initialization, data shuffling, and MLM masking are seeded for
reproducibility. GPU execution is not guaranteed to be bitwise deterministic,
so repeated runs may differ slightly.

### Plotting

Plot training curves after their runs finish:

```bash
python training_curves/plot.py results/mlm-training-from-scratch
python training_curves/plot.py results/mlm-training-checkpoint
python training_curves/plot.py results/rna-seq-training-from-scratch
python training_curves/plot.py results/rna-seq-training-checkpoint
```

# LARES
This repository contains the code for the paper: **LARES: Targeted Lateral Movement Detection in Evolving Networks Through Source Host Identification**.

## Quick start on Google Colab

[`colab/LARES_artifact.ipynb`](colab/LARES_artifact.ipynb) reproduces the paper's main
results end to end on a free Colab runtime. Open it with **File → Upload notebook** in
[Google Colab](https://colab.research.google.com/), or through **File → Open notebook →
GitHub**, set `DATASET_URL` in section 1 to a Google Drive link of
`lanl_optc_datasets_compiled.tar.gz` (0.6 GB, sha256
`a0b9cd2c95557061ffbc66daaad9b8a88f8820c1ac38a6127474bc57f7756981`), then
**Runtime → Run all**.

The notebook clones this repository with its released weights, downloads the compiled
graph snapshots, evaluates from the weights, and prints the reproduced numbers next to
the published ones (Table II, Table VII, Figure 2, and the *Detection* row of Table III).
About an hour on a CPU runtime; no GPU required. Interrupted sessions can be resumed by
running all cells again: finished steps are detected and skipped. Pointing `DATASET_URL`
at the raw 14.35 GB dataset also works; the notebook then compiles the snapshots itself,
which takes several hours.

## Installation

### Install dependencies
Install the conda env. Set `CUDA_TAG` accordingly.

```shell
conda create -n lares python=3.9 -y
conda activate lares

# CUDA_TAG: e.g. cu102 | cu113 | cu118 | cu121 | cpu
CUDA_TAG=cu118
TORCH_VER=2.2.2

if [ "$CUDA_TAG" = "cpu" ]; then
  pip install torch==$TORCH_VER --index-url https://download.pytorch.org/whl/cpu
else
  pip install torch==$TORCH_VER --index-url https://download.pytorch.org/whl/$CUDA_TAG
fi

PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}%2B${CUDA_TAG}.html"
pip install torch_sparse torch_scatter torch_cluster pyg_lib torch_geometric -f $PYG_URL

pip install pandas==2.0.3 scikit-learn==1.0.2 matplotlib==3.7.4 igraph==0.11.3 wandb==0.15.11
```

`torch_sparse`, `torch_scatter`, `torch_cluster` and `pyg_lib` are optional: no module
under `src/` imports them, and plain `torch_geometric` is sufficient. Skipping them avoids
a long source build on platforms without prebuilt wheels. A pip-only dependency list is
also provided in [`requirements.txt`](requirements.txt).

Tested with Python 3.9 through 3.12.

## Dataset
We made available our preprocessed LANL and OpTC datasets. Within each dataset, one file represents a 1-min TW in csv format. To save place when the archive is unzipped, the csv files for the OpTC dataset are compressed in `.gz` format and the data loader directly reads from the compressed csv file, thereby saving disk space. Once downloaded, the preprocessed datasets have to be **compiled** from 1-min csv files to 30-min (OpTC) and 60-min (LANL) graph snapshots in PyTorch tensor format saved as `.pkl` files.
Once compiled, the graphs can be loaded in an efficient way, and experiments can be reproduced.

### Download datasets

- Download `lanl_optc_datasets.tar.gz` (14.35 GB, 18.44 GB uncompressed) from Google Drive
  with [this link](GOOGLE-DRIVE-LINK-TO-RAW-DATASET)
  - or from the CLI with:
  ```
  pip install gdown
  gdown --fuzzy "GOOGLE-DRIVE-LINK-TO-RAW-DATASET"
  ```
- Decompress the archive in the root of the repo with `tar -xzf lanl_optc_datasets.tar.gz`
- If your uncompressed folder is elsewhere, point the `LARES_DATA_ROOT` environment
  variable at the absolute path of the `lanl_optc_datasets/` folder, e.g.
  `export LARES_DATA_ROOT=/data/lanl_optc_datasets`. Editing `ROOT` in
  `src/utils/config.py` also works.

> Google Drive throttles files that were downloaded heavily in the past 24 hours. The
> Colab notebook does not depend on this archive: it downloads the much smaller compiled
> snapshots instead (see [Quick start on Google Colab](#quick-start-on-google-colab)).

### Compile datasets

To compile the graphs in a usable tensor format, simply run `datasets.py` followed by the **dataset** name (name of the folder where the compiled graphs will be stored on disk) and the inductive experiment to apply on these graphs. The compiled graphs will be generated within the `lanl_optc_datasets` folder set in `ROOT`. Note that these commands may be run in parallel. If `dataset_name` is changed, please ensure to change it accordingly within `config.py`. Setting `LARES_COMPRESS_COMPILED=1` writes the compiled snapshots as gzip-compressed pickles (several times smaller, loaded transparently); the compiled archive distributed with the Colab notebook was produced this way.

> The hosts masked by `Exp1`-`Exp3` are drawn with a fixed seed, so compilation is
> deterministic on a given Python version. The draw itself differs between Python 3.10
> and earlier and Python 3.11 and later, because `random.sample` no longer accepts a
> set. This changes which hosts are hidden during training, not the test split, so
> evaluation from the released weights is unaffected.

LANL
```
python src/datasets.py  --dataset=LANL  --dataset_name=LANL_exp0 --inductive_experiment=Exp0
python src/datasets.py  --dataset=LANL  --dataset_name=LANL_exp1 --inductive_experiment=Exp1
python src/datasets.py  --dataset=LANL  --dataset_name=LANL_exp2 --inductive_experiment=Exp2
python src/datasets.py  --dataset=LANL  --dataset_name=LANL_exp3 --inductive_experiment=Exp3
```

OPTC
```
python src/datasets.py  --dataset=OPTC  --dataset_name=OPTC_exp0 --inductive_experiment=Exp0
python src/datasets.py  --dataset=OPTC  --dataset_name=OPTC_exp1 --inductive_experiment=Exp1
python src/datasets.py  --dataset=OPTC  --dataset_name=OPTC_exp2 --inductive_experiment=Exp2
python src/datasets.py  --dataset=OPTC  --dataset_name=OPTC_exp3 --inductive_experiment=Exp3
```

## Reproduce experiments
### Node and edge level results

Once the datasets compiled and the dependencies installed, the detection results can be run reproduced either by loading the weights or by training from scratch.

#### From weights

LANL
```
python src/main.py --config=LANL_inductive_exp0 --use_weights=True
python src/main.py --config=LANL_inductive_exp1 --use_weights=True
python src/main.py --config=LANL_inductive_exp2 --use_weights=True
python src/main.py --config=LANL_inductive_exp3 --use_weights=True
```

OPTC
```
python src/main.py --config=OPTC_inductive_exp0 --use_weights=True
python src/main.py --config=OPTC_inductive_exp1 --use_weights=True
python src/main.py --config=OPTC_inductive_exp2 --use_weights=True
python src/main.py --config=OPTC_inductive_exp3 --use_weights=True
```

#### From training

LANL
```
python src/main.py --config=LANL_inductive_exp0
python src/main.py --config=LANL_inductive_exp1
python src/main.py --config=LANL_inductive_exp2
python src/main.py --config=LANL_inductive_exp3
```

OPTC
```
python src/main.py --config=OPTC_inductive_exp0
python src/main.py --config=OPTC_inductive_exp1
python src/main.py --config=OPTC_inductive_exp2
python src/main.py --config=OPTC_inductive_exp3
```

### MCC @ 10-100% of unseen hosts

1. Create datasets for each experiment 

You need to substitute first: `{10|20|30|40|50|60|70|80|90}`

LANL
```
# Included
python src/datasets.py --dataset=LANL --dataset_name=LANL_include_{10|20|30|40|50|60|70|80|90} --inductive_experiment=custom

# Excluded
python src/datasets.py --dataset=LANL --dataset_name=LANL_exclude_{10|20|30|40|50|60|70|80|90} --inductive_experiment=custom
```

OpTC
```
# Included
python src/datasets.py --dataset=OPTC --dataset_name=OPTC_include_{10|20|30|40|50|60|70|80|90} --inductive_experiment=custom

# Excluded
python src/datasets.py --dataset=OPTC --dataset_name=OPTC_exclude_{10|20|30|40|50|60|70|80|90} --inductive_experiment=custom
```

2. Run experiments



LANL
```
# Included
python src/main.py --config=LANL_custom --dataset_name=LANL_include_10
python src/main.py --config=LANL_custom --dataset_name=LANL_include_20
python src/main.py --config=LANL_custom --dataset_name=LANL_include_30
python src/main.py --config=LANL_custom --dataset_name=LANL_include_40
python src/main.py --config=LANL_custom --dataset_name=LANL_include_50
python src/main.py --config=LANL_custom --dataset_name=LANL_include_60
python src/main.py --config=LANL_custom --dataset_name=LANL_include_70
python src/main.py --config=LANL_custom --dataset_name=LANL_include_80
python src/main.py --config=LANL_custom --dataset_name=LANL_include_80

# Excluded
python src/main.py --config=LANL_custom --dataset_name=LANL_exclude_10
python src/main.py --config=LANL_custom --dataset_name=LANL_exclude_20
python src/main.py --config=LANL_custom --dataset_name=LANL_exclude_30
python src/main.py --config=LANL_custom --dataset_name=LANL_exclude_40
python src/main.py --config=LANL_custom --dataset_name=LANL_exclude_50
python src/main.py --config=LANL_custom --dataset_name=LANL_exclude_60
python src/main.py --config=LANL_custom --dataset_name=LANL_exclude_70
python src/main.py --config=LANL_custom --dataset_name=LANL_exclude_80
python src/main.py --config=LANL_custom --dataset_name=LANL_exclude_80
```

OpTC
```
# Included
python src/main.py --config=OPTC_custom --dataset_name=OPTC_include_{10|20|30|40|50|60|70|80|90}

# Excluded
python src/main.py --config=OPTC_custom --dataset_name=OPTC_exclude_{10|20|30|40|50|60|70|80|90}
```

### Ablation study

Substitute `{0|1|2|3}` first.

LANL
```
python src/main.py --config=LANL_inductive_exp{0|1|2|3} --encoder=EULER
python src/main.py --config=LANL_inductive_exp{0|1|2|3} --encoder=Argus_LANL
python src/main.py --config=LANL_inductive_exp{0|1|2|3} --encoder=GIN
python src/main.py --config=LANL_inductive_exp{0|1|2|3} --use_recurrent=True
python src/main.py --config=LANL_inductive_exp{0|1|2|3} --use_recon_loss=False --use_contra_loss=True 
python src/main.py --config=LANL_inductive_exp{0|1|2|3} --use_recon_loss=False --use_ap_loss=True
python src/main.py --config=LANL_inductive_exp{0|1|2|3} --use_direct_edge_detection=True
python src/main.py --config=LANL_inductive_exp{0|1|2|3} --use_skip_connection=False
python src/main.py --config=LANL_inductive_exp{0|1|2|3} --node_features=ones
python src/main.py --config=LANL_inductive_exp{0|1|2|3}
```

OPTC
```
python src/main.py --config=OPTC_inductive_exp{0|1|2|3} --encoder=EULER
python src/main.py --config=OPTC_inductive_exp{0|1|2|3} --encoder=Argus_OPTC
python src/main.py --config=OPTC_inductive_exp{0|1|2|3} --encoder=GIN
python src/main.py --config=OPTC_inductive_exp{0|1|2|3} --use_recurrent=True
python src/main.py --config=OPTC_inductive_exp{0|1|2|3} --use_recon_loss=False --use_contra_loss=True 
python src/main.py --config=OPTC_inductive_exp{0|1|2|3} --use_recon_loss=False --use_ap_loss=True
python src/main.py --config=OPTC_inductive_exp{0|1|2|3} --use_direct_edge_detection=True
python src/main.py --config=OPTC_inductive_exp{0|1|2|3} --use_skip_connection=False
python src/main.py --config=OPTC_inductive_exp{0|1|2|3} --node_features=ones
python src/main.py --config=OPTC_inductive_exp{0|1|2|3}
```

### Hyperparameter changes

Reproduce the hyperparameter robustness experiments (same for LANL) as follows. Snapshot size requires to re-create datasets with the new size directly in the code.

LANL
```
python src/main.py --config=LANL_inductive_exp2 --out_node_emb=32
python src/main.py --config=LANL_inductive_exp2 --out_node_emb=64
python src/main.py --config=LANL_inductive_exp2 --out_node_emb=128

python src/main.py --config=LANL_inductive_exp2 --lr=0.01
python src/main.py --config=LANL_inductive_exp2 --lr=0.001
python src/main.py --config=LANL_inductive_exp2 --lr=0.0001
```

OpTC
```
python src/main.py --config=OPTC_inductive_exp2 --out_node_emb=32
python src/main.py --config=OPTC_inductive_exp2 --out_node_emb=64
python src/main.py --config=OPTC_inductive_exp2 --out_node_emb=128

python src/main.py --config=OPTC_inductive_exp2 --lr=0.01
python src/main.py --config=OPTC_inductive_exp2 --lr=0.001
python src/main.py --config=OPTC_inductive_exp2 --lr=0.0001
```

## Artifact evaluation (ACSAC 2026)

**Public release.** The full artifact is released publicly: this repository (source code
and trained weights), the preprocessed LANL and OpTC datasets with the ground-truth labels
used in the paper, and the Colab notebook. No component is withheld after evaluation.

**Requirements.** Linux or macOS, Python 3.9–3.13, ~4 GB of RAM (a CPU-only run is
sufficient to evaluate from the released weights; a GPU only speeds it up). Disk: ~2 GB
for the weights-based path, >40 GB for the full pipeline.

**Where each claim is reproduced.**

| Paper item | Command or notebook section | Needs training |
|---|---|---|
| Table II, LARES rows | `python src/main.py --config={LANL,OPTC}_inductive_exp{0,1,2,3} --use_weights=True` | no |
| Table VII, LARES rows | same commands, node-level block of the output | no |
| Figure 2, FP counts on LANL | TP/FP counts of the Exp0 run above (baseline counts from §V-B) | no |
| Table III, *Detection* row | add `--use_direct_edge_detection=True` | no |
| Table III, other rows | *Ablation study* section above | yes |
| Figures 1 and 4 | *MCC @ 10-100% of unseen hosts* section above | yes |
| Figure 8 | *Hyperparameter changes* section above | yes |

Each command in the *From weights* section runs in a few minutes and prints the node-level
metrics (stage 1, Table VII) followed by the edge-level metrics (stage 2, Table II). The
Colab notebook runs the eight Table II commands and compares each metric against the
published values.

## License

See [license](LICENSE).

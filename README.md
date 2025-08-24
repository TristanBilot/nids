# FAUCON
This repository contains the code for the paper: **FAUCON: Targeted Lateral Movement Detection in Evolving Networks Through Source Host Identification**.

## Installation

### Clone the repo

```shell
git clone https://github.com/TristanBilot/faucon.git
cd faucon
```

### Install dependencies
Install the conda env. Set `CUDA_TAG` accordingly.

```shell
conda create -n faucon python=3.9 -y
conda activate faucon

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

## Dataset
We made available our preprocessed LANL and OpTC datasets. Within each dataset, one file represents a 1-min TW in csv format. To save place when the archive is unzipped, the csv files for the OpTC dataset are compressed in `.gz` format and the data loader directly reads from the compressed csv file, thereby saving disk space. Once downloaded, the preprocessed datasets have to be **compiled** from 1-min csv files to 30-min (OpTC) and 60-min (LANL) graph snapshots in PyTorch tensor format saved as `.pkl` files.
Once compiled, the graphs can be loaded in an efficient way, and experiments can be reproduced.

### Download datasets

- Download `focus_datasets.zip` with [this link](https://mega.nz/file/eBZRxRJJ#IvzPBKITBL3_TjjJemeiJDPE5p3TpJQ0Fnmq7WNWLXo) (14.35 GB zipped, 18.44 GB unzipped)
  - or from CLI with:
  ```
  sudo apt install megatools
  megadl https://mega.nz/file/eBZRxRJJ#IvzPBKITBL3_TjjJemeiJDPE5p3TpJQ0Fnmq7WNWLXo
  ```
- Decompress the archive with `unzip focus_datasets.zip`
- The default path to this folder is `faucon/focus_datasets/`. If your folder is elsewhere, set the variable `ROOT` to the absolute path to the unzipped `focus_datasets/` folder `src/utils/config.py`.

### Compile datasets

To compile the graphs in a usable tensor format, simply run `datasets.py` followed by the **dataset** name (name of the folder where the compiled graphs will be stored on disk) and the inductive experiment to apply on these graphs. The compiled graphs will be generated within the `focus_datasets` folder set in `ROOT`. Note that these commands may be run in parallel. If `dataset_name` is changed, please ensure to change it accordingly within `config.py`.

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

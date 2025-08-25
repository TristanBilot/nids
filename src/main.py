import argparse
import copy
import os
import sys
import random
from pathlib import Path
from pprint import pprint
from time import perf_counter

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import wandb

from src.models.model import Model
from torch.optim import Adam
from utils.config import *
from utils.torch_device import COMPUTE_DEVICE as DEVICE
from utils.eval import (
    MetricAverager,
    NodeEvaluator,
    NodeEvaluator,
)
from utils.utils import (
    preprocess_args,
    replace_with_best_config,
)
from datasets import get_loaders

# These are default arguments.
# Actual arguments are in config.py
config = {
    "use_wandb": "False",
    "use_verbose": "False",
    "iterations": 2,
    "epochs": 20,
    "dataset": "LANL",
    "use_weights": "False",
    "config": "None",
    "lr": 0.0,
    "weight_decay": 0.000001,
    "grad_accumulation": 1,
    "gnn_flow": "target_to_source",
    "node_dropout": 0.25,
    "hid_node_emb": -1,
    "out_node_emb": -1,
    "use_recurrent": "",
    "use_vae": "False",
    "use_no_self_loops": "True",
    "use_skip_connection": "",
    "use_fast_neg_sampling": "False",
    "encoder": "FAUCON",
    "use_recon_loss": "",
    "use_contra_loss": "",
    "use_ap_loss": "",
    "lstm_or_gru": "lstm",
    "use_only_autoencoder": "False",
    "keep_only_features": "all",
    "node_features": "ones",
    "dataset_name": "",
    "inductive_experiment": "None",
    "use_without_tanh": "",
    "use_with_epsilon": "",
    "ablation": "",  # [encoder | decoder | pretraining | rnn]
    "seed": 0,
    "alpha": -1,
    "beta": -1,
    "use_direct_edge_detection": "",
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=config["dataset"], help="Name of the dataset (OPTC | LANL)")
parser.add_argument("--use_weights", type=str, default=config["use_weights"], help="If True, loads the weights for the experiment instead of training")
parser.add_argument("--config", type=str, default=config["config"], help="One of the config variable key available in `src/config/` -> BEST_CONFIGS (e.g. LANL_inductive_exp1)")
parser.add_argument("--iterations", type=int, default=config["iterations"], help="Number of iterations/experiments to compute the mean/std of experiments")
parser.add_argument("--epochs", type=int, default=config["epochs"], help="Training epochs")
parser.add_argument("--lr", type=float, default=config["lr"], help="Fine tuning/actual training lr")
parser.add_argument("--weight_decay", type=float, default=config["weight_decay"], help="To avoid too large gradients")
parser.add_argument("--grad_accumulation", type=int, default=config["grad_accumulation"], help="Number of batches/snapshot to wait before backprop")
parser.add_argument("--seed", type=int, default=config["seed"], help="Seed to reproduce experiments")
parser.add_argument("--use_wandb", type=str, default=config["use_wandb"], help="Set to False unless you use Weights & Biases")
parser.add_argument("--use_verbose", type=str, default=config["use_verbose"], help="Whether to print details about FPs")
parser.add_argument("--gnn_flow", type=str, default=config["gnn_flow"], help="target_to_source or source_to_target GNN flow")
parser.add_argument("--node_dropout", type=float, default=config["node_dropout"], help="Dropout rate with node features dropping")
parser.add_argument("--hid_node_emb", type=int, default=config["hid_node_emb"], help="Hidden node embedding size used in GNN and RNN layers")
parser.add_argument("--out_node_emb", type=int, default=config["out_node_emb"], help="Output node embedding size")
parser.add_argument("--use_recurrent", type=str, default=config["use_recurrent"], help="Whether to use a RNN")
parser.add_argument("--use_vae", type=str, default=config["use_vae"], help="Whether to use a VAE instead of a classical AE")
parser.add_argument("--use_no_self_loops", type=str, default=config["use_no_self_loops"], help="Whether to use self-loops in the input graphs")
parser.add_argument("--use_skip_connection", type=str, default=config["use_skip_connection"], help="Whether to add skip connection with edge features to the edge embeddings")
parser.add_argument("--use_fast_neg_sampling", type=str, default=config["use_fast_neg_sampling"], help="Whether to use the fast neg sampling instead of the harder same src different dst neg sampling")
parser.add_argument("--encoder", type=str, default=config["encoder"], help="Name of the encoder")
parser.add_argument("--use_recon_loss", type=str, default=config["use_recon_loss"], help="Whether to use the reconstruction-based edge decoding")
parser.add_argument("--use_contra_loss", type=str, default=config["use_contra_loss"], help="Whether to use the contrastive-based edge decoding")
parser.add_argument("--use_ap_loss", type=str, default=config["use_ap_loss"], help="Whether to use the AP loss as in ARGUS")
parser.add_argument("--lstm_or_gru", type=str, default=config["lstm_or_gru"], help="Whether to use LSTM of GRU as recurrent layer (LSTM | GRU)")
parser.add_argument("--use_only_autoencoder", type=str, default=config["use_only_autoencoder"], help="Whether to only use edge decoder")
parser.add_argument("--keep_only_features", type=str, default=config["keep_only_features"], help="The edge features to keep in the encoding (all | nb_edges)")
parser.add_argument("--node_features", type=str, default=config["node_features"], help="The node features to keep in the encoding (one_hot | ones)")
parser.add_argument("--dataset_name", type=str, default=config["dataset_name"], help="The path to the folder where ready-to-use datasets are saved")
parser.add_argument("--inductive_experiment", type=str, default=config["inductive_experiment"], help="Name of the inductive experiment to simulate (Exp1 | Exp2 | Exp3)")
parser.add_argument("--ablation", type=str, default=config["ablation"], help="Name of the ablation to experiment (encoder | decoder | pretraining | rnn)")
parser.add_argument("--alpha", type=str, default=config["alpha"], help="Controls the range of nodes to consider in the kmeans")
parser.add_argument("--beta", type=str, default=config["beta"], help="Controls the range of edges to consider in the attack reconstruction")
parser.add_argument("--use_direct_edge_detection", type=str, default=config["use_direct_edge_detection"], help="Use EULER's and ARGUS' threhsold")

# Args used for the encoder ablation study
parser.add_argument("--use_without_tanh", type=str, default=config["use_without_tanh"], help="Model improvement explained in the paper")
parser.add_argument("--use_with_epsilon", type=str, default=config["use_with_epsilon"], help="Model improvement explained in the paper")

args = parser.parse_args()
args = replace_with_best_config(args, config, BEST_CONFIGS)
pprint(args.__dict__)

args = preprocess_args(args, config)

wandb.init(
    mode="offline" if args.use_wandb else "disabled",
    project="wandb_project",
    config=config,
)

current_path = os.path.dirname(os.path.abspath(__file__))
weights_path = "/".join(current_path.split("/")[:-1]) + "/weights/"
weight_file = os.path.join(weights_path, f"weights_{args.config}.pkl")
os.makedirs(weights_path, exist_ok=True)
conf = CONF(args)


# For reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def eval_model(model, valid_loader):
    model.eval()
    tot_loss = 0.0
    all_losses = []

    with torch.no_grad():
        for i, data in enumerate(valid_loader.iterate_lazily()):
            data = data.to(DEVICE)
            losses = model.inference(data)
            tot_loss += losses.mean().item()
            all_losses.extend(losses.tolist())

    return tot_loss / valid_loader.nb_batches


def main(
    loaders: list,
    args=None,
):
    torch.cuda.reset_peak_memory_stats(DEVICE)
    train_loader, valid_loader, test_loader = loaders
    best_metric, best_stats, best_model = -1, 0, None
    epoch_time, val_loss = 0, 0

    model = Model(
        x_dim=conf["NODE_FEAT_SIZE"],
        nb_nodes=conf["NB_NODES"],
        nb_edge_feats=conf["EDGE_FEAT_SIZE"],
        device=DEVICE,
        gnn_flow=args.gnn_flow,
        node_dropout=args.node_dropout,
        hid_node_emb=args.hid_node_emb,
        out_node_emb=args.out_node_emb,
        use_recurrent=args.use_recurrent,
        use_vae=args.use_vae,
        use_skip_connection=args.use_skip_connection,
        encoder=args.encoder,
        use_recon_loss=args.use_recon_loss,
        use_contra_loss=args.use_contra_loss,
        lstm_or_gru=args.lstm_or_gru,
        use_only_autoencoder=args.use_only_autoencoder,
        dataset=args.dataset,
        use_fast_neg_sampling=args.use_fast_neg_sampling,
        use_without_tanh=args.use_without_tanh,
        use_with_epsilon=args.use_with_epsilon,
        use_ap_loss=args.use_ap_loss,
    ).to(DEVICE)

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    node_tester = NodeEvaluator(model, test_loader, args, conf["NB_NODES"], args.alpha, args.beta, args.use_direct_edge_detection, args.dataset)
    
    for e in range(args.epochs):
        
        if args.use_weights:
            model.load_state_dict(torch.load(weight_file))
            
        else:
            print(f"\nEPOCH {e+1}/{args.epochs}")
            opt.zero_grad()
            model.reset()

            loss_acc = torch.tensor(0.0, device=DEVICE)
            times = []
            for i, data in enumerate(train_loader.iterate_lazily()):
                data = data.to(DEVICE)

                model.train()

                start = perf_counter()
                loss = model(data)
                loss_acc += loss

                # Backward pass
                if (
                    i + 1
                ) % args.grad_accumulation == 0 or i == train_loader.nb_batches - 1:
                    loss_acc.backward()
                    # Avoid too large gradients
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    opt.step()
                    opt.zero_grad()
                    model.detach()
                    loss_acc = torch.tensor(0.0, device=DEVICE)
                    
                end = perf_counter()
                times.append(end - start)

            # End of epoch
            epoch_time = np.sum(times)

            # Validation
            opt.zero_grad()
            val_loss = eval_model(model, valid_loader)
            print(
                f"Train Loss: {loss.item():.7f}, Epoch Time: {epoch_time}, lr: {opt.param_groups[0]['lr']}"
            )

        # Test
        print("Evaluation on test set:")
        node_stats = node_tester.evaluate(path=f"logs/{args.config}")

        n_auc, precision, recall, tpr, fpr, f1, ap, mcc = node_stats

        stats = {
            "epoch": e,
            "epoch_time": epoch_time,
            "val_loss": val_loss,
            "precision": precision,
            "recall": recall,
            "tpr": tpr,
            "fpr": fpr,
            "f1": f1,
            "mcc": mcc,
        }
        
        wandb.log(stats)
        peak_memory = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)  # Convert to GB
        print(f"Peak CUDA memory usage: {peak_memory:.2f} GB")

        if mcc > best_metric:
            best_metric = mcc
            best_stats = stats
            best_model = copy.deepcopy(model.state_dict())
            
        if args.use_weights:
            break

    print(f"Best stats: {best_stats}")
    print(f"\nBest Metrics:")
    print(f"Recall: {best_stats['recall']}")
    print(f"Precision: {best_stats['precision']}")
    print(f"MCC: {best_stats['mcc']}")
    
    # torch.save(
    #     best_model,
    #     weight_file,
    # )
    
    return model, best_stats


if __name__ == "__main__":
    loaders = get_loaders(args)
    metric_averager = MetricAverager()

    for it in range(args.iterations):

        trained_model, best_stats = main(
            loaders=loaders,
            args=args,
        )

        if best_stats != 0:
            metric_averager.add(best_stats)

    if args.iterations > 1:
        print(f"Iteration {it+1}/{args.iterations}: Mean of best stats:")
        best_stats_mean = metric_averager.get_avg_std()
        pprint(best_stats_mean)
        wandb.log(best_stats_mean)

    wandb.finish()

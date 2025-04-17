import os
from collections import defaultdict
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
)
from sklearn.metrics import average_precision_score as ap_score
from utils.torch_device import COMPUTE_DEVICE as DEVICE
from sklearn.cluster import KMeans


class AbstractEvaluator:
    def __init__(self, model, test_loader, config):
        self.model = model
        self.test_loader = test_loader
        self.config = config

    def evaluate(self):
        raise NotImplementedError("Abstract method.")

    def compute_metrics(self, ys, y_hats, scoress, file_name, i):
        attack_idxs = (ys == 1).nonzero()[0]

        overall_acc = (ys == y_hats).mean()
        precision, recall, f1, support = precision_recall_fscore_support(
            ys, y_hats, average="binary"
        )

        tp = y_hats[ys == 1].sum()
        fp = y_hats[ys == 0].sum()
        tpr = y_hats[ys == 1].mean()
        fpr = y_hats[ys == 0].mean()

        len_positives = len((ys == 1).nonzero()[0])
        len_negatives = len((ys == 0).nonzero()[0])
        
        mcc = matthews_corrcoef(ys, y_hats)

        if scoress is not None:
            try:
                auc = roc_auc_score(ys, scoress)
            except:
                auc = float("nan")
            try:
                ap = ap_score(ys, scoress)
            except:
                ap = float("nan")
        else:
            auc = float("nan")
            ap = float("nan")

        prompt = f"""Evaluation: found {tp} attacks / {len(attack_idxs)} attack samples ({(tp/len(attack_idxs))*100:.5f}%).\n""" \
            f"""f"TPR: {tpr:.3f} | FPR: {fpr:.6f} | Overall acc: {overall_acc:.6f} | AP: {ap:.3f} | F1: {f1:.3f} | AUC: {auc:.3f} | recall: {recall:.3f} | precision: {precision:.3f} | MCC: {mcc:.2f}\n""" \
            f"""TP: {tp}/{len_positives} | FP: {fp}/{len_negatives}\n"""
        print(prompt)

        with open(os.path.join(file_name, f"logs_{i}.log"), 'a') as file:
            file.write(prompt)
        
        return auc, precision, recall, tpr, fpr, f1, ap, mcc

class NodeEvaluator(AbstractEvaluator):
    def __init__(self, model, test_loader, config, nb_nodes, alpha, beta, use_direct_edge_detection, dataset):
        super().__init__(model, test_loader, config)
        self.nb_nodes = nb_nodes
        self.alpha = alpha
        self.beta = beta
        self.i = 0
        self.use_direct_edge_detection = use_direct_edge_detection
        self.dataset = dataset

    def evaluate(self, path):
        self.model.eval()

        shift = 0
        all_node_ys = torch.zeros((self.nb_nodes,))
        all_edge_ys = []
        all_edge_index = []
        all_edge_losses = []
        node_to_max_loss = {}
        times = []
        
        with torch.no_grad():
            for i, data in enumerate(self.test_loader.iterate_lazily()):
                data = data.to(DEVICE)
                start = perf_counter()
                losses = self.model.inference(data)
                end = perf_counter()
                times.append(end - start)

                # Compute node labels.
                attack_src_nodes = data.edge_index[0, data.y == 1]
                
                all_node_ys[attack_src_nodes] = 1
                
                all_edge_ys.append(data.y)
                all_edge_index.append(data.edge_index)
                all_edge_losses.append(losses)
                
                max_values = torch.zeros((self.nb_nodes,), dtype=torch.float, device=losses.device)
                max_values = max_values.scatter_reduce(0, data.edge_index[0], losses, reduce="amax")
                
                for node, max_loss in enumerate(max_values):
                    node_to_max_loss[node] = max(node_to_max_loss.get(node, 0), max_loss.item())
            
            print(f"\nMean inference runtime: {np.mean(times):.2f}s")
            print(f"Total inference runtime: {np.sum(times):.2f}s\n")
            
            node_ys, node_losses, nodes = [], [], []
            for node, max_loss in node_to_max_loss.items():
                node_ys.append(all_node_ys[node].item())
                node_losses.append(max_loss)
                nodes.append(node)
                
            all_edge_index = torch.cat(all_edge_index, dim=1).detach().cpu()
            all_edge_ys = torch.cat(all_edge_ys).detach().cpu()
            all_edge_losses = torch.cat(all_edge_losses).detach().cpu()
            
            node_y_hats_indices = compute_kmeans_labels(torch.tensor(node_losses), self.alpha) # predicted malicious nodes
            node_yhats = np.zeros((self.nb_nodes,))
            node_yhats[node_y_hats_indices] = 1
            
            print_scores(node_losses, node_ys, nodes)
            
            print("\nNode detection metrics:")
            os.makedirs(path, exist_ok=True)
            node_stats = self.compute_metrics(np.array(node_ys), np.array(node_yhats), node_losses, path, self.i)
            
            if self.use_direct_edge_detection:
                print("\nStandard edge-level detection:")
                fixed_recall = 0.56 if self.dataset == "LANL" else 0.48 # for fair comparison we fix recall
                edge_yhats = y_hat_at_recall(np.array(all_edge_ys), all_edge_losses, fixed_recall)
                edge_stats = self.compute_metrics(np.array(all_edge_ys), edge_yhats, all_edge_losses, path, self.i)
            
            else:
                print("\nEdge detection metrics:")
                edge_y_hats_indices = get_top_k_outgoing_edges(np.array(all_edge_index), np.array(all_edge_losses), node_y_hats_indices, self.beta)
                edge_yhats = np.zeros((len(all_edge_ys),))
                edge_yhats[edge_y_hats_indices] = 1
                edge_stats = self.compute_metrics(np.array(all_edge_ys), edge_yhats, all_edge_losses, path, self.i)
            
            self.i += 1
            return edge_stats

def compute_kmeans_labels(results, topk_K):
    score_values, sorted_indices = torch.sort(results)
    score_values = score_values.numpy()
    sorted_indices = sorted_indices.numpy()

    last_N_scores = score_values[-topk_K:]

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans.fit(last_N_scores.reshape(-1, 1))

    centroids = kmeans.cluster_centers_.flatten()
    highest_cluster_index = np.argmax(centroids)

    best_cluster_indices = np.where(kmeans.labels_ == highest_cluster_index)[0]
    best_cluster_indices = np.concatenate([[best_cluster_indices[0] - 1], best_cluster_indices])

    indices = []
    for idx in best_cluster_indices:
        global_idx = len(score_values) - topk_K + idx
        indices.append(global_idx)
        
    return sorted_indices[np.array(indices)][::-1]

def get_top_k_outgoing_edges(edge_index, edge_scores, top_k_nodes, k):
    top_k_edge_indices = []

    for node_id in top_k_nodes:
        outgoing_edges = np.where(edge_index[0] == node_id)[0]

        if outgoing_edges.size > 0:
            outgoing_scores = edge_scores[outgoing_edges]
            top_k_outgoing_indices = outgoing_edges[np.argsort(outgoing_scores)[-k:][::-1]]
            top_k_edge_indices.extend(top_k_outgoing_indices)

    return top_k_edge_indices

def y_hat_at_recall(y_truth, scores, target_recall):
    y_truth = np.array(y_truth)
    scores = np.array(scores)

    # Sort scores and labels by descending score
    desc_score_indices = np.argsort(-scores)
    y_truth_sorted = y_truth[desc_score_indices]
    scores_sorted = scores[desc_score_indices]

    # Cumulative true positives and false positives
    tp = np.cumsum(y_truth_sorted)
    fn = tp[-1] - tp
    recall = tp / (tp + fn + 1e-10)

    # Find index where recall >= target_recall
    valid_indices = np.where(recall >= target_recall)[0]
    if len(valid_indices) == 0:
        return np.zeros_like(y_truth)  # All 0s: recall never reached

    # Threshold corresponding to first valid index
    threshold_index = valid_indices[0]
    threshold = scores_sorted[threshold_index]

    # Generate y_hat based on threshold
    y_hat = (scores >= threshold).astype(int)
    return y_hat

def print_scores(scores, y_truth, nodes):
    scores_0 = [score for score, label in zip(scores, y_truth) if label == 0]
    scores_1 = [score for score, label in zip(scores, y_truth) if label == 1]
    
    nodes_1 = [node for node, label in zip(nodes, y_truth) if label == 1]
    
    fp_counts = []
    for score in scores_1:
        fp_count = sum(s >= score for s in scores_0)
        fp_counts.append(fp_count)

    # Print the results in a nicely formatted way
    output_string = "Malicious nodes' score | Number of FPs\n"
    output_string += "-----------------|---------------\n"
    for score, fp_count, node in zip(scores_1, fp_counts, nodes_1):
        output_string += f"{node}: {score:<18} | {fp_count}\n"
    print(output_string)

class MetricAverager:
    def __init__(self):
        self.storage = defaultdict(list)

    def add(self, metrics_dict):
        for k, v in metrics_dict.items():
            self.storage[k].append(v)

    def get_avg_std(self):
        best_precision_idx = np.argmax(self.storage["precision"])
        return {
            k: {
                "mean": np.mean(v),
                "std": np.std(v),
                "best": v[best_precision_idx],
            }
            for k, v in self.storage.items()
        }

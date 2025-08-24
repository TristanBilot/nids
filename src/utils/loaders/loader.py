import glob
import math
import os
import pickle
import random
import re
from collections import defaultdict
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data
from tqdm import tqdm

from ..config import *


class AbstractLoader:
    def __init__(
        self,
        device,
        ranges: tuple = None,
        nb_processes: int = NB_PROCESSES,
        preprocessed_path: str = None,
        compiled_path: str = None,
        do_compile: bool = None,
        nb_nodes: str = None,
        batch_size: int = None,
        use_gzip: bool = False,
        compress_edges: bool = True,
        use_no_self_loops: bool = True,
        keep_only_features: str = "all",
        node_features: str = "one_hot",
        inductive_experiment: str = "None",
        malicious_src_nodes: list = None,
        use_flows: bool = False,
        dataset_name: str = None,
        **kwargs,
    ):
        if None in [ranges, preprocessed_path, compiled_path, do_compile, nb_nodes]:
            raise ValueError("A required value is set to None.")

        self.device = device
        self.preprocessed_path = preprocessed_path
        self._batch_count = 0
        self._nb_nodes = nb_nodes
        self._nb_processes = nb_processes
        self._batch_size = batch_size
        self._use_gzip = use_gzip
        self._use_no_self_loops = use_no_self_loops
        self._do_compile = do_compile
        self._keep_only_features = keep_only_features
        self._node_features = node_features
        self._inductive_experiment = inductive_experiment
        self._malicious_src_nodes = malicious_src_nodes

        self._start_range_initial = ranges[0]
        self._end_range_initial = ranges[1]

        self._compress_edges = compress_edges
        self._malicious_hosts_cache = defaultdict(set)

        self._dataset_name = dataset_name
        self._inductive_nodes = self._compute_inductive_nodes()
        self._eye = None
        self._use_flows = use_flows
        self._compiled_path = os.path.join(compiled_path, dataset_name)

    def iterate_lazily(
        self, random_shift: int = 0
    ) -> Generator[Tuple[pd.DataFrame, np.ndarray, np.ndarray], None, None]:
        """
        Yields a generator that returns a dataframe along with edge
        and node features for a single batch of size `batch_size`.

        Returns:
            Generator[
                dataframe
                edge features
                node features
            ]
        """
        self._all_sorted_snapshots = self._get_all_snapshots()
        self._nb_batches = self._get_nb_batches()

        for batch in tqdm(range(self._nb_batches), total=self._nb_batches):
            if self._do_compile:
                snap = self._load_batch_multiprocessing()
            else:
                snap = self._load_preprocessed_snapshot()

            if len(snap.edge_index) == 0:
                continue

            if self._use_no_self_loops:
                snap = self._remove_self_loops(snap)

            snap.edge_attr = self._keep_only_some_features(snap.edge_attr)
            
            if self._eye == None:
                if self._node_features == "one_hot":
                    self._eye = torch.eye(self._nb_nodes, device=self.device)
                else:
                    self._eye = torch.ones((self._nb_nodes, self._nb_nodes), device=self.device)
            snap.x = self._eye

            yield snap

        self.reset_to_start()

    def _load_batch_multiprocessing(
        self,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Loads one batch of the OpTC dataset into memory.
        By default, returns in TGN input format, as shown above.

        Returns:
            dataframe
            edge features
            node features
        """
        start = self._batch_count * self._batch_size
        end = min(
            (self._batch_count + 1) * self._batch_size, len(self._all_sorted_snapshots)
        )
        workers = self._nb_processes

        # Only gets snapshots from the current snapshot.
        nb_snapshots = len(self._all_sorted_snapshots[start:end])

        # We don't want more workers than the number of snapshots to process.
        workers = min(workers, nb_snapshots)

        # Assign a number of snapshot to each worker.
        jobs_per_worker = [nb_snapshots // workers] * workers

        # Add remaining jobs to latter workers.
        remainder = nb_snapshots % workers
        for w in range(remainder):
            jobs_per_worker[workers - 1 - w] += 1

        # Arguments for each worker.
        worker_snapshots = []
        for w in range(workers):
            upto = min(start + jobs_per_worker[w], end)

            snaps = self._all_sorted_snapshots[start:upto]
            worker_snapshots.append(snaps)
            start += jobs_per_worker[w]

        # Run the workers with multiprocessing.
        results = Parallel(n_jobs=workers, prefer="processes")(
            delayed(self._load_chunk_of_snapshots)(
                worker_snapshots[i],
            )
            for i in range(workers)
        )

        all_edge_index = [r[0] for r in results]
        all_edge_feats = [r[1] for r in results]
        all_labels = [r[2] for r in results]

        # Merge the results from all workers.
        edge_index, edge_feats, labels = self._merge_workers_output(
            all_edge_index, all_edge_feats, all_labels
        )

        data = self._to_torch_geo_data(edge_index, edge_feats, labels)

        self._batch_count += 1
        
        if self._inductive_experiment != "None":
            data = self._apply_inductive_experiment(data)
        data = self._set_root_attack_nodes(data)

        if self._do_compile:
            self._save_preprocessed_snapshot(data.edge_index, data.edge_attr, data.y)
        
        return data

    def _load_chunk_of_snapshots(
        self, files: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loop over the snapshots from a single worker and returns
        the concatenation of these snapshots.

        Args:
            files: the files on which to loop on.

        Returns:
            dataframe
            edge features
        """
        # LANL
        if self._use_flows:

            all_df_adjs, all_edge_feats, all_flows = [], [], []
            for file in files:
                df_adj, edge_feats, flows = self._load_one_snapshot(file)
                all_df_adjs.append(df_adj)
                all_edge_feats.append(edge_feats)
                all_flows.append(flows)

            # Concatenate the dataframes from chunk.
            df_adj = pd.concat(all_df_adjs, ignore_index=True)

            # Concatenate edge features in the same way.
            edge_feats = np.concatenate((all_edge_feats), axis=0)

            # Concat flows
            flows = pd.concat(all_flows, ignore_index=True)

            if self._compress_edges:
                return self._compress_graph(df_adj, edge_feats, flows)
            else:
                raise NotImplementedError("Need implementation for non-compressed graphs.")

            return (
                df_adj,
                edge_feats,
                flows,
            )
        
        # OpTC
        all_df_adjs, all_edge_feats = [], []
        for file in files:
            df_adj, edge_feats = self._load_one_snapshot(file)
            all_df_adjs.append(df_adj)
            all_edge_feats.append(edge_feats)

        # Concatenate the dataframes from chunk.
        df_adj = pd.concat(all_df_adjs, ignore_index=True)

        # Concatenate edge features in the same way.
        edge_feats = np.concatenate((all_edge_feats), axis=0)

        if self._compress_edges:
            return self._compress_graph(df_adj, edge_feats)
        else:
            raise NotImplementedError("Need implementation for non-compressed graphs.")

        return (
            df_adj,
            edge_feats,
        )

    def _load_one_snapshot(self, file: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Creates input data structures for a single snapshot.

        Returns:
            List[
                DataFrame[src, dst, ts, y, idx],
                np.array[np.array] (edge feats),
        """

        raise NotImplementedError("Abstract method.")

    def _preprocess_one_snapshot(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        All preprocessing operations to perform on each snapshot should be located
        in this function.

        Args:
            df: the raw input dataframe from the OpTC dataset.

        Returns:
            dataframe: the preprocessed dataframe that will be loaded in the batch.
        """

        raise NotImplementedError("Abstract method.")

    def _merge_workers_output(
        self, all_edge_index, all_edge_feats, all_labels
    ):
        """
        Concatenates the result of all workers.

        Args:
            all_df_adjs: array of dataframes
            all_edge_feats: array of edge feat arrays
        """

        # Concatenate edge features in the same way.
        edge_index = np.concatenate((all_edge_index), axis=0)
        edge_feats = np.concatenate((all_edge_feats), axis=0)
        labels = np.concatenate((all_labels), axis=0)

        return (
            edge_index,
            edge_feats,
            labels,
        )

    def _get_all_snapshots(self) -> List[str]:
        """
        Returns the list of all the paths to snapshots for the current time range.
        """
        # All existing snapshots are collected, and sorted by timestamp.
        extension = ".csv.gz" if self._use_gzip else ".csv"

        # Select all snapshot files present in the folder.
        preprocessed_path = os.path.join(self.preprocessed_path, f"*{extension}")
        all_snapshots = glob.glob(preprocessed_path)
        if len(all_snapshots) == 0:
            raise FileNotFoundError(f"No files found in {preprocessed_path}")

        get_sc_from_filename = lambda x: int(re.search(r"(\d+)(?=\.csv(?:\.gz)?$)", x).group(1))
        all_sorted_snapshots = sorted(
            all_snapshots, key=lambda x: get_sc_from_filename(x)
        )

        start_index = [
            i
            for i, sc in enumerate(all_sorted_snapshots)
            if get_sc_from_filename(sc) == self._start_range_initial
        ]

        end_index = [
            i
            for i, sc in enumerate(all_sorted_snapshots)
            if get_sc_from_filename(sc) == self._end_range_initial
        ]

        assert (
            len(start_index) == 1
        ), f"The starting snapshot {self._start_range_initial} does not exist in files."
        assert (
            len(end_index) == 1
        ), f"The end snapshot {self._end_range_initial} does not exist in files."
        start_index = start_index[0]
        end_index = end_index[0]

        snapshots = all_sorted_snapshots[start_index : end_index + 1]

        return snapshots

    def _get_nb_batches(self) -> int:
        """
        Returns the number of batch to process the current range of data.
        Some snapshots don't have any data (no existing file), so we need
        to count the number of existing snapshots to compute the number of batches.
        """
        nb_batches = math.ceil(len(self._all_sorted_snapshots) / self._batch_size)
        return nb_batches

    def _compress_graph(self, df, edge_feats):
        raise NotImplementedError("Abstract method.")

    def _standardize_edge_feats(self, edge_feats):
        """For flow features that are not present in all edges, or for categorical features with
        0s, we want to compute the statistics only on the non-0 values. We also want to update
        just the non-0 features.
        """
        nonzero = edge_feats.nonzero()[0]
        if len(nonzero) == 0:
            return edge_feats

        e = 1e-6
        std = edge_feats[nonzero].std()

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        standardized = (edge_feats[nonzero] - edge_feats[nonzero].mean()) / (std + e)
        standardized = sigmoid(standardized)

        edge_feats[nonzero] = standardized
        return edge_feats

    def saved_snapshot_path_at_idx(self, idx):
        return os.path.join(self._compiled_path, f"{idx}.pkl")

    def _save_preprocessed_snapshot(self, edge_index, edge_feats, labels):
        os.makedirs(self._compiled_path, exist_ok=True)

        with open(self.saved_snapshot_path_at_idx(self._batch_count), "wb") as f:
            pickle.dump((edge_index, edge_feats, labels), f)

    def _load_preprocessed_snapshot(self):
        self._batch_count += 1
        with open(self.saved_snapshot_path_at_idx(self._batch_count), "rb") as f:
            if self._eye == None:
                self._eye = torch.eye(self._nb_nodes, device=self.device)

            (edge_index, edge_feats, labels, *_) = pickle.load(f)
            # We generate the node feats at the fly here.
            x = self._eye

            data = self._to_torch_geo_data(edge_index, edge_feats, labels, x)

        return data

    def _to_torch_geo_data(self, edge_index, edge_feats, labels, x=None) -> Data:
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        if not isinstance(edge_feats, torch.Tensor):
            edge_feats = torch.tensor(edge_feats, dtype=torch.float)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_feats, y=labels)

    def _remove_self_loops(self, data):
        indices = (data.edge_index[0] != data.edge_index[1]).nonzero().squeeze()

        return Data(
            x=data.x,
            edge_index=data.edge_index[:, indices],
            edge_attr=data.edge_attr[indices],
            y=data.y[indices],
        )

    def _concat_preprocessed_snapshots(self, datas):
        edge_indices = torch.cat(datas["edge_index"], dim=1)
        edge_attrs = torch.cat(datas["attr"])
        ys = torch.cat(datas["y"])

        edge_to_feats = defaultdict(
            lambda: {
                "edge_attr": torch.zeros_like(edge_attrs[0]),
                "y": 0,
            }
        )

        for u, v, attr, y in zip(
            edge_indices[0].tolist(), edge_indices[1].tolist(), edge_attrs, ys.tolist()
        ):
            edge = (u, v)
            edge_to_feats[edge]["edge_attr"] = edge_to_feats[edge]["edge_attr"] + attr
            edge_to_feats[edge]["y"] = max(edge_to_feats[edge]["y"], y)

        # convert to np arrays
        edge_index, labels, e_feats = [], [], []
        for edge, feats in edge_to_feats.items():
            edge_index.append(edge)
            e_feats.append(feats["edge_attr"].numpy())
            labels.append(feats["y"])

        edge_index, e_feats, labels = (
            np.array(edge_index),
            np.array(e_feats),
            np.array(labels),
        )
        edge_index = np.array([edge_index[:, 0], edge_index[:, 1]])

        # Standardize all features along the 1-axis.
        for i in range(len(e_feats[0])):
            e_feats[:, i] = self._standardize_edge_feats(e_feats[:, i])

        n_feats = torch.eye(self._nb_nodes)
        return self._to_torch_geo_data(edge_index, e_feats, n_feats, labels)

    @property
    def nb_batches(self):
        return self._nb_batches

    def reset_to_start(self):
        self._batch_count = 0

    def _set_root_attack_nodes(self, data):
        indices = torch.nonzero(torch.isin(data.edge_index[0, :], torch.tensor(self._malicious_src_nodes))).squeeze()
        y = torch.zeros((data.y.shape[0],), dtype=data.y.dtype)
        y[indices] = data.y[indices]
        data.y = y
        return data

    def _compute_inductive_nodes(self):
        if self._inductive_experiment == "None":
            return None

        # Transductive experiment, returns all nodes
        if self._inductive_experiment == "Exp0":
            return torch.tensor([])
        
        # Removes 30% of nodes with their connections, without any malicious nodes inside.
        if self._inductive_experiment == "Exp1":
            return self.mask_percent_nodes_include_malicious(p=0.3)

        # Removes 30% of nodes with their connections, including all malicious nodes
        if self._inductive_experiment == "Exp2":
            return self.mask_percent_nodes_exclude_malicious(p=0.3)

        # Removes 50% of nodes with their connections, including all malicious nodes
        if self._inductive_experiment == "Exp3":
            return self.mask_percent_nodes_exclude_malicious(p=0.5)
        
        if self._inductive_experiment == "custom":
            split = self._dataset_name.split("_")
            exclude, p = split[1], split[2]
            p = float(int(p) / 100)
            if exclude == "include":
                return self.mask_percent_nodes_include_malicious(p=p)
            elif exclude == "exclude":
                return self.mask_percent_nodes_exclude_malicious(p=p)
            raise ValueError(f"Invalid format {exclude}")

        raise ValueError("Invalid experiment name.")
    
    def mask_percent_nodes_include_malicious(self, p: float):
        percent = int(self._nb_nodes * p)

        available_nodes = set(range(1, self._nb_nodes)) - set(
            self._malicious_src_nodes
        )
        sample = torch.tensor(random.sample(available_nodes, percent))

        return sample
    
    def mask_percent_nodes_exclude_malicious(self, p: float):
        percent = int(self._nb_nodes * p)

        available_nodes = set(range(1, self._nb_nodes))
        sample = torch.tensor(random.sample(available_nodes, percent))
        sample[: len(self._malicious_src_nodes)] = torch.tensor(
            self._malicious_src_nodes
        )

        return sample
    
    def _apply_inductive_experiment(self, data):
        assert self._inductive_nodes != None, "Need to compute inductive nodes."

        ei, e_attr, y = data.edge_index, data.edge_attr, data.y
        sample = self._inductive_nodes

        mask = (ei[0].unsqueeze(1) == sample).any(1) | (
            ei[1].unsqueeze(1) == sample
        ).any(1)
        keep_edges = ~mask

        ei = ei[:, keep_edges]
        e_attr = e_attr[keep_edges, :]
        y = y[keep_edges]

        return Data(x=data.x, edge_index=ei, edge_attr=e_attr, y=y)

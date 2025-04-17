import os
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from ..config import *
from .loader import AbstractLoader


class OpTCLoader(AbstractLoader):
    def __init__(
        self,
        ranges: tuple,
        compiled_path: str,
        dataset_name: str,
        preprocessed_path: str = PREPROCESSED_PATH,
        do_compile: bool = True,
        nb_nodes: str = NB_NODES,
        filter_flow: str = "start",
        filter_noise_protocols: bool = False,
        normalize: bool = False,
        batch_size: int = COMPUTED_SNAPSHOT_SIZE,
        malicious_src_nodes: list = ROOT_ATTACK_NODES,
        use_gzip: bool = True,
        **kwargs,
    ):
        super().__init__(
            ranges=ranges,
            compiled_path=compiled_path,
            do_compile=do_compile,
            preprocessed_path=preprocessed_path,
            nb_nodes=nb_nodes,
            batch_size=batch_size,
            malicious_src_nodes=malicious_src_nodes,
            use_gzip=use_gzip,
            use_flows=False,
            dataset_name=dataset_name,
            **kwargs,
        )

        assert filter_flow in [
            "all",
            "start",
            "message",
        ], f"Filtering mode {filter_flow} unknown."
        self._filter_flow = filter_flow
        self._filter_noise_protocols = filter_noise_protocols
        self._normalize = normalize

    def _load_one_snapshot(self, file: str) -> Tuple[pd.DataFrame, np.ndarray]:
        if not os.path.isfile(file):
            print(f"Warning: file not found: {file}")
            return

        # If gzip compression, we read the csv while decompressing it.
        compression = "gzip" if self._use_gzip else "infer"

        # Reads the csv and assign an int to each column index.
        df = pd.read_csv(
            file,
            index_col=False,
            header=None,
            names=range(TS, IDX + 1),
            dtype={TS: str},
            compression=compression,
        )

        # Additional preprocessing, normalisation, ...
        df = self._preprocess_one_snapshot(df)

        # Output schema.
        df_adj = df[[SRC, DST, TS, LABEL, IDX]]
        edge_feats = df[[SIZE, PROTOCOL, SRC_PORT, DST_PORT]].to_numpy()
        # edge_feats = np.array(range(len(df_adj)))

        return (
            df_adj,
            edge_feats,
        )

    def _preprocess_one_snapshot(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._filter_flow == "start":
            df = df[df[ACT] == "F_S"]
        elif self._filter_flow == "message":
            df = df[df[ACT] == "F_M"]

        if self._filter_noise_protocols:
            df = df.drop(df[(df[SRC_PORT] == 137) & (df[DST_PORT] == 137)].index)
            df = df.drop(df[(df[SRC_PORT] == 138) & (df[DST_PORT] == 138)].index)

            df = df.drop(df[(df[DST_PORT] == 5355)].index)

        # Normalization raises an unwanted warning.
        pd.set_option("mode.chained_assignment", None)

        # Normalization of numeric features. (only when no compress is used)
        if self._normalize:
            df[PROTOCOL] = df[PROTOCOL].apply(
                lambda x: (x - PROTO_MIN) / (PROTO_MAX - PROTO_MIN)
            )

            if self._filter_flow == "message":
                df[SIZE] = df[SIZE].apply(
                    lambda x: (x - SIZE_MIN) / (SIZE_MAX - SIZE_MIN)
                )

        # Normalize timestamps similarly as in the dataset from TGN:
        df[TS] = pd.to_numeric(df[TS])
        df[TS] = df[TS].apply(lambda x: x - FIRST_TS)

        df.fillna(0, inplace=True)

        return df

    def _compress_graph(self, df, edge_feats):
        df_adj = df[[SRC, DST, LABEL]]

        nb_e_feats = EDGE_FEAT_SIZE  # (nb_duplicate_edges, protos, s_ports, d_ports, is_self_loop)
        edge_to_feats = defaultdict(lambda: np.zeros((nb_e_feats,)))
        edge_to_labels = defaultdict(int)
        edge_to_count = defaultdict(int)

        df_adj = df_adj.to_dict()

        for i, (src, dst, y) in enumerate(
            zip(df_adj[SRC].values(), df_adj[DST].values(), df_adj[LABEL].values())
        ):
            edge = (src, dst)
            proto, s_port, d_port = (
                int(edge_feats[i][1]),
                int(edge_feats[i][2]),
                int(edge_feats[i][3]),
            )

            if proto == 6:
                edge_to_feats[edge][1] += 1
            elif proto == 17:
                edge_to_feats[edge][2] += 1
            elif proto == 1:
                edge_to_feats[edge][3] += 1
            elif proto == 58:
                edge_to_feats[edge][4] += 1

            if s_port == 137:
                edge_to_feats[edge][5] += 1
            elif s_port == 138:
                edge_to_feats[edge][6] += 1
            elif s_port == 68:
                edge_to_feats[edge][7] += 1
            elif s_port == 135:
                edge_to_feats[edge][8] += 1
            elif s_port == 136:
                edge_to_feats[edge][9] += 1
            elif s_port == 0:
                edge_to_feats[edge][10] += 1
            elif s_port == 8:
                edge_to_feats[edge][11] += 1
            elif s_port == 5355:
                edge_to_feats[edge][12] += 1
            elif 5355 < s_port <= 50_000:
                edge_to_feats[edge][13] += 1
            elif 50_000 < s_port <= 60_000:
                edge_to_feats[edge][14] += 1
            elif 60_000 < s_port:
                edge_to_feats[edge][15] += 1

            if d_port == 137:
                edge_to_feats[edge][16] += 1
            elif d_port == 138:
                edge_to_feats[edge][17] += 1
            elif d_port == 67:
                edge_to_feats[edge][18] += 1
            elif d_port == 135:
                edge_to_feats[edge][19] += 1
            elif d_port == 136:
                edge_to_feats[edge][20] += 1
            elif d_port == 0:
                edge_to_feats[edge][21] += 1
            elif d_port == 8:
                edge_to_feats[edge][22] += 1
            elif d_port == 139:
                edge_to_feats[edge][23] += 1
            elif d_port == 1900:
                edge_to_feats[edge][24] += 1
            elif d_port == 5355:
                edge_to_feats[edge][25] += 1
            elif 5355 < d_port <= 50_000:
                edge_to_feats[edge][26] += 1
            elif 50_000 < d_port <= 60_000:
                edge_to_feats[edge][27] += 1
            elif 60_000 < d_port:
                edge_to_feats[edge][28] += 1

            edge_to_labels[edge] = max(edge_to_labels[edge], y)
            edge_to_count[edge] += 1

        # convert to np arrays
        edge_index, labels, e_feats = [], [], []
        for edge, feats in edge_to_feats.items():
            feats[0] = edge_to_count[
                edge
            ]  # add the total number of edges between two nodes.
            edge_index.append(edge)
            e_feats.append(feats)
            labels.append(edge_to_labels[edge])

        edge_index, e_feats, labels = (
            np.array(edge_index),
            np.array(e_feats),
            np.array(labels),
        )
        # edge_index = edge_index.reshape(2, -1)
        if len(edge_index) > 0:
            edge_index = np.array([edge_index[:, 0], edge_index[:, 1]])

            # Standardize all features along the 1-axis.
            for i in range(nb_e_feats):
                e_feats[:, i] = self._standardize_edge_feats(e_feats[:, i])

        # Add self-loops.
        # uniq_nodes = list(set(edge_index[0]) | set(edge_index[1]))
        # self_loops = np.array([uniq_nodes, uniq_nodes])

        # #   Dummy edge feature for added self-loops.
        # self_loop_e_feats = np.zeros((len(self_loops[0]), nb_e_feats))
        # self_loop_e_feats[:, -1] = 1

        # edge_index = np.concatenate([edge_index, self_loops], axis=1)
        # e_feats = np.concatenate([e_feats, self_loop_e_feats], axis=0)
        # labels = np.concatenate([labels, np.zeros(len(self_loops[0],), dtype=int)])

        return (
            edge_index,
            e_feats,
            labels,
        )

    def _keep_only_some_features(self, e_feats):
        features = self._keep_only_features.split(",")
        to_keep = (np if isinstance(e_feats, np.ndarray) else torch).zeros_like(e_feats)

        for feature in features:
            if feature == "all":
                to_keep[:, :] = e_feats[:, :]
            elif feature == "nb_edges":
                to_keep[:, 0] = e_feats[:, 0]
            elif feature == "protos":
                to_keep[:, 1:5] = e_feats[:, 1:5]
            elif feature == "src_ports":
                to_keep[:, 5:16] = e_feats[:, 5:16]
            elif feature == "dst_ports":
                to_keep[:, 16:29] = e_feats[:, 16:29]
            else:
                raise ValueError("Invalid feature name.")

        return to_keep

import os
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from ..config import *
from .loader import AbstractLoader


class LANLLoader(AbstractLoader):
    def __init__(
        self,
        ranges: tuple,
        compiled_path: str,
        dataset_name: str,
        preprocessed_path: str = LANL_PREPROCESSED_AUTH_PATH,
        do_compile: bool = True,
        nb_nodes: str = LANL_NB_NODES,
        batch_size: int = LANL_COMPUTED_SNAPSHOT_SIZE,
        malicious_src_nodes: list = LANL_ROOT_ATTACK_NODES,
        use_gzip: bool = False,
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
            use_flows=True,
            dataset_name=dataset_name,
            **kwargs,
        )

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
            names=range(LANL_TS, LANL_LABEL + 1),
            dtype={
                LANL_TS: str,
                LANL_SRC: int,
                LANL_DST: int,
                LANL_SRC_USER: int,
                LANL_DST_USER: int,
                LANL_AUTH_TYPE: int,
                LANL_LOGON_TYPE: int,
                LANL_AUTH_ORIENT: int,
                LANL_SUCCESS: int,
                LANL_LABEL: int,
            },
            compression=compression,
        )

        # Additional preprocessing, normalisation, ...
        df = self._preprocess_one_snapshot(df)

        # Output schema.
        df_adj = df[
            [LANL_SRC, LANL_DST, LANL_TS, LANL_LABEL, LANL_SRC_USER, LANL_DST_USER]
        ]
        edge_feats = df[
            [LANL_SUCCESS, LANL_LOGON_TYPE, LANL_AUTH_TYPE, LANL_AUTH_ORIENT]
        ].to_numpy()


        # For each auth file, we get the corresponding flow file.
        flow_file = os.path.join(LANL_PREPROCESSED_FLOWS_PATH, file.split("/")[-1].replace("graph", "flows"))

        # Only read non-empty snapshots to avoid error.
        if os.path.exists(flow_file) and os.path.getsize(flow_file) > 0:
            flow_df = pd.read_csv(
                flow_file,
                index_col=False,
                header=None,
                names=range(LANL_FLOW_TS, LANL_FLOW_LABEL + 1),
                dtype={
                    LANL_FLOW_TS: int,
                    LANL_FLOW_SRC: int,
                    LANL_FLOW_DST: int,
                    LANL_FLOW_SRC_PORT: int,
                    LANL_FLOW_DST_PORT: int,
                    LANL_FLOW_PROTO: int,
                    LANL_FLOW_DUR: int,
                    LANL_FLOW_PKT_COUNT: int,
                    LANL_FLOW_BYTE_COUNT: int,
                    LANL_FLOW_LABEL: int,
                },
                compression=compression,
            )
        else:
            flow_df = pd.DataFrame()

        return (
            df_adj,
            edge_feats,
            flow_df,
        )

    def _preprocess_one_snapshot(self, df: pd.DataFrame) -> pd.DataFrame:
        # Removes new test nodes from test set to test inductivity.
        # if self._nodes_to_ignore != None:
        #     df = df[~df[SRC].isin(self._nodes_to_ignore)]
        #     df = df[~df[DST].isin(self._nodes_to_ignore)]

        # Normalization raises an unwanted warning.
        pd.set_option("mode.chained_assignment", None)

        # Normalize timestamps similarly as in the dataset from TGN:
        df[TS] = pd.to_numeric(df[TS])

        df.fillna(0, inplace=True)

        return df

    def _compress_graph(self, df, edge_feats, df_flows):
        df_adj = df[[LANL_SRC, LANL_DST, LANL_LABEL]]

        NUM_AUTH_FEATURES = 6
        NUM_FLOW_FEATURES = 7
        nb_e_feats = NUM_AUTH_FEATURES + NUM_FLOW_FEATURES

        edge_to_feats = defaultdict(lambda: np.zeros((nb_e_feats,)))
        edge_to_labels = defaultdict(int)
        edge_to_auth_count = defaultdict(int)
        edge_to_flow_count = defaultdict(int)

        # Flow features (but only present in 2% of edges)
        edge_to_pkt_count = defaultdict(list)
        edge_to_byte_count = defaultdict(list)
        edge_to_duration = defaultdict(list)

        df_adj = df_adj.to_dict()

        # Auth
        for i, (src, dst, y) in enumerate(
            zip(
                df_adj[LANL_SRC].values(),
                df_adj[LANL_DST].values(),
                df_adj[LANL_LABEL].values(),
            )
        ):
            edge = (src, dst)
            is_success = int(edge_feats[i][0])

            # Sucess/failure
            if is_success:
                edge_to_feats[edge][1] += 1
            else:
                edge_to_feats[edge][2] += 1

            # Source user type
            if edge_feats[i][1] == 1:
                edge_to_feats[edge][3] += 1
            elif edge_feats[i][1] == 2:
                edge_to_feats[edge][4] += 1
            elif edge_feats[i][1] == 3:
                edge_to_feats[edge][5] += 1

            edge_to_labels[edge] = max(edge_to_labels[edge], y)
            edge_to_auth_count[edge] += 1

        # Flows
        df_flows = df_flows.to_dict()
        if len(df_flows) > 0:
            for i, (src, dst, dur, pkt_count, byte_count) in enumerate(
                zip(
                    df_flows[LANL_FLOW_SRC].values(),
                    df_flows[LANL_FLOW_DST].values(),
                    df_flows[LANL_FLOW_DUR].values(),
                    df_flows[LANL_FLOW_PKT_COUNT].values(),
                    df_flows[LANL_FLOW_BYTE_COUNT].values(),
                )
            ):
                edge = (src, dst)
                edge_to_flow_count[edge] += 1
                edge_to_pkt_count[edge].append(pkt_count)
                edge_to_byte_count[edge].append(byte_count)
                edge_to_duration[edge].append(dur)

        def mean_or_zero(x):
            return 0.0 if len(x) == 0 else np.mean(x)

        def std_or_zero(x):
            return 0.0 if len(x) == 0 else np.std(x)

        # convert to np arrays
        edge_index, labels, e_feats = [], [], []
        for edge, feats in edge_to_feats.items():
            feats[0] = edge_to_auth_count[
                edge
            ]  # add the total number of edges between two nodes.
            feats[6] = edge_to_flow_count[edge]  # add the total number of flows.
            feats[7] = mean_or_zero(
                edge_to_duration[edge]
            )  # add the mean of duration, as in argus.
            feats[8] = std_or_zero(
                edge_to_duration[edge]
            )  # add the std of duration, as in argus.
            feats[9] = mean_or_zero(
                edge_to_pkt_count[edge]
            )  # add the mean of packet count, as in argus.
            feats[10] = std_or_zero(
                edge_to_pkt_count[edge]
            )  # add the std of packet count, as in argus.
            feats[11] = mean_or_zero(
                edge_to_byte_count[edge]
            )  # add the mean of byte count, as in argus.
            feats[12] = std_or_zero(
                edge_to_byte_count[edge]
            )  # add the std of byte count, as in argus.

            edge_index.append(edge)
            e_feats.append(feats)
            labels.append(edge_to_labels[edge])

        edge_index, e_feats, labels = (
            np.array(edge_index),
            np.array(e_feats),
            np.array(labels),
        )
        edge_index = np.array([edge_index[:, 0], edge_index[:, 1]])

        # Standardize all features along the 1-axis.
        e_feats = standardize_euler_argus(e_feats)

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
            elif feature == "success_auth":
                to_keep[:, 1:3] = e_feats[:, 1:3]
            elif feature == "src_user_type":
                to_keep[:, 3:6] = e_feats[:, 3:6]
            elif feature == "flow_features":
                to_keep[:, 6:13] = e_feats[:, 6:13]
            else:
                raise ValueError("Invalid feature name.")

        return to_keep

def standardize_euler_argus(edge_features: np.ndarray) -> np.ndarray:
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    res = np.zeros(edge_features.shape)
    for i in range(edge_features.shape[1]):
        x = edge_features[:, i]
        x = x.astype(np.float32)
        x = (x.astype(np.int64) / (x.std() + 1e-6)).astype(np.int64)
        x = sigmoid(x)
        res[:, i] = x
    return res

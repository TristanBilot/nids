"""
This file centralizes all the configuration variables for preprocessing.
"""
import os


# NOTE: if needed, replace with the absolute path to the uncompressed datasets folder
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "focus_datasets")

# Root paths
OPTC_ROOT = os.path.join(ROOT, "OPTC/")
SPLIT = OPTC_ROOT + "split/"

# Required files
NODE_MAP_PATH = SPLIT + "nmap.pkl"  # Should be created before preprocessing
LABELS_CSV = OPTC_ROOT + "labels.csv"  # Should be downloaded online

# Output paths
OUT_NETWORK_PATH = OPTC_ROOT + "network/"
OUT_HOST_PATH = SPLIT + "host/"

# Logs
LOG_FILE = OPTC_ROOT + "logs.txt"
PREPROCESS_SNASHOTS_LOG_FILE = OPTC_ROOT + "preproc_snapshots_logs.txt"

# Preprocessed dataset paths
PREPROCESSED_PATH = OPTC_ROOT + "preprocessed/"

TRAIN_PATH = OPTC_ROOT + "compiled/train/"
VALID_PATH = OPTC_ROOT + "compiled/valid/"
TEST_PATH = OPTC_ROOT + "compiled/test/"

# Compute
NB_PROCESSES = 1
SNAPSHOT_SIZE = 60  # seconds
MAX_BUFFER = 2**16  # Max lines workers store before flushing out

COMPUTED_SNAPSHOT_SIZE = 30  # nb files per snapshot
NB_NODES = 1114  # in the whole dataset (from EULER paper)
NODE_FEAT_SIZE = NB_NODES  # one-hot encoding of the node ID as node feat
EDGE_FEAT_SIZE = 30

# From the labels.csv file.
FIRST_FLOW_LABEL_SNAPSHOT = 9700
LAST_FLOW_LABEL_SNAPSHOT = 11024

# From the preprocessed snapshot files.
FIRST_SNAPSHOT = 0
LAST_SNAPSHOT = 12671

END_TRAIN = 3101
END_VALID = FIRST_FLOW_LABEL_SNAPSHOT - 10
END_TEST = LAST_FLOW_LABEL_SNAPSHOT + 10

# OpTC-specific timestamps
# Ranges of snapshots (in minutes) for each set.
TIME_RANGES = {
    "TRAIN": (FIRST_SNAPSHOT, 8500),
    "VALID": (8500 + 1, 9500),  # same as in Euler
    "TEST": (9500 + 1, LAST_SNAPSHOT,),  # we consider the range from first to last anomalous flow
    "FIND_LABELS": (8845 + 25 * 30, LAST_SNAPSHOT),  # just for experiments
    "ALL": (FIRST_SNAPSHOT, LAST_SNAPSHOT),
    "BYPASS_TRAIN": (FIRST_SNAPSHOT, FIRST_SNAPSHOT + 100),
    "BYPASS_TEST": (END_TRAIN + 1, END_TRAIN + 100),
}
# From checking the first/last line of each file
FIRST_TS = 1568676405.62800
LAST_TS = 1569436694.309

# Input schema of one network flow OpTC event.
TS = 0
ACT = 1
LABEL = 2
SRC = 3
DST = 4
SRC_PORT = 5
DST_PORT = 6
PROTOCOL = 7
SIZE = 8
IDX = 9

HOST_COLUMNS = ["t", "a", "y", "h", "s", "d", "pcp", "path", "new_path", "size"]
HOST_COLUMNS_W_INDEX = HOST_COLUMNS + ["idx"]

NETWORK_COLUMNS = ["t", "a", "y", "s", "d", "s_port", "d_port", "proto", "size"]
NETWORK_COLUMNS_W_INDEX = NETWORK_COLUMNS + ["idx"]

# OpTC feature statistics (for normalization)
S_PORT_MIN = 135
S_PORT_MAX = 65535

D_PORT_MIN = 135
D_PORT_MAX = 65532

PROTO_MIN = 6
PROTO_MAX = 17

SIZE_MIN = 4
SIZE_MAX = 491419

# Labels source nodes distribution: {201: 2}, {402: 1}, {501: 56}
ROOT_ATTACK_NODES = [201, 402, 501]


# LANL VARIABLES
LANL_ROOT = os.path.join(ROOT, "LANL/")
LANL_AUTH_FILE = LANL_ROOT + "raw/auth.txt.gz"
LANL_RED_FILE = LANL_ROOT + "raw/redteam.txt.gz"
LANL_FLOWS_FILE = LANL_ROOT + "raw/flows.txt.gz"

LANL_PREPROCESSED_AUTH_PATH = LANL_ROOT + "preprocessed/auth/"
LANL_PREPROCESSED_FLOWS_PATH = LANL_ROOT + "preprocessed/flows/"

LANL_COMPILED_TRAIN_PATH = LANL_ROOT + "compiled/train/"
LANL_COMPILED_VALID_PATH = LANL_ROOT + "compiled/valid/"
LANL_COMPILED_TEST_PATH = LANL_ROOT + "compiled/test/"

LANL_COMPUTED_SNAPSHOT_SIZE = 60  # 1h snap like in Argus for LANL
LANL_TOTAL_NB_NODES = (
    17685  # Max number over the whole dataset (but we only use first 14 days)
)
LANL_NB_NODES = 13184  # Max node for the first 14 days
LANL_NODE_INDUCTIVE_FEAT_SIZE = 9
LANL_EDGE_FEAT_SIZE = 13

LANL_FIRST_ATTACK = 150885  # time in seconds
LANL_FIRST_ATTACK_FILE = 2513
LANL_NB_FILES = 83518

# Dataset ranges: the +7 is to round snapshots to get exactly
# snapshots of 60 files, and no less.
LANL_TRAIN_END = LANL_FIRST_ATTACK_FILE - 3 * LANL_COMPUTED_SNAPSHOT_SIZE + 7 - 1
LANL_VALID_END = LANL_FIRST_ATTACK_FILE - LANL_COMPUTED_SNAPSHOT_SIZE + 7 - 1
LANL_TEST_END = 20160 - 1  # 13 full days
# LANL_TEST_END = 21600 -1 # Full 14-days as in Argus (buggy as we need to recompute LANL_NB_NODES (larger))

LANL_TIME_RANGES = {
    "TRAIN": (0, LANL_TRAIN_END),
    "VALID": (LANL_TRAIN_END + 1, LANL_VALID_END),
    "TEST": (LANL_VALID_END + 1, LANL_TEST_END),
}

LANL_ROOT_ATTACK_NODES = [9660]

# Preprocessed auth csv file fields
LANL_TS = 0
LANL_SRC = 1
LANL_DST = 2
LANL_SRC_USER = 3
LANL_DST_USER = 4
LANL_AUTH_TYPE = 5
LANL_LOGON_TYPE = 6
LANL_AUTH_ORIENT = 7
LANL_SUCCESS = 8
LANL_LABEL = 9

# Flow file fields
LANL_FLOW_TS = 0
LANL_FLOW_SRC = 1
LANL_FLOW_DST = 2
LANL_FLOW_SRC_PORT = 3
LANL_FLOW_DST_PORT = 4
LANL_FLOW_PROTO = 5
LANL_FLOW_DUR = 6
LANL_FLOW_PKT_COUNT = 7
LANL_FLOW_BYTE_COUNT = 8
LANL_FLOW_LABEL = 9


def CONF(args):
    LANL_NODE_FEAT_SIZE = LANL_NB_NODES

    conf = {
        "LANL": {
            "PREPROCESSED_PATH": LANL_PREPROCESSED_AUTH_PATH,
            "TIME_RANGES": LANL_TIME_RANGES,
            "NB_NODES": LANL_NB_NODES,
            "NODE_FEAT_SIZE": LANL_NODE_FEAT_SIZE,
            "EDGE_FEAT_SIZE": LANL_EDGE_FEAT_SIZE,
            "TRAIN_PATH": LANL_COMPILED_TRAIN_PATH,
            "VALID_PATH": LANL_COMPILED_VALID_PATH,
            "TEST_PATH": LANL_COMPILED_TEST_PATH,
            "BATCH_SIZE": LANL_COMPUTED_SNAPSHOT_SIZE,
            "MALICIOUS_NODES": ROOT_ATTACK_NODES,
        },
        "OPTC": {
            "PREPROCESSED_PATH": PREPROCESSED_PATH,
            "TIME_RANGES": TIME_RANGES,
            "NB_NODES": NB_NODES,
            "NODE_FEAT_SIZE": NODE_FEAT_SIZE,
            "EDGE_FEAT_SIZE": EDGE_FEAT_SIZE,
            "TRAIN_PATH": TRAIN_PATH,
            "VALID_PATH": VALID_PATH,
            "TEST_PATH": TEST_PATH,
            "BATCH_SIZE": COMPUTED_SNAPSHOT_SIZE,
            "MALICIOUS_NODES": LANL_ROOT_ATTACK_NODES,
        },
    }
    return conf[args.dataset]


CONFIGS = {
    "LANL_base": {
        "epochs": 10,
        "iterations": 1,
        "dataset": "LANL",
        "encoder": "FAUCON",
        "lr": 0.0001,
        "alpha": 3,
        "beta": 350,
        "use_contra_loss": "False",
        "use_recon_loss": "True",
        "use_skip_connection": "True",
        "use_recurrent": "False",
        "keep_only_features": "all",
        "out_node_emb": 32,
        "hid_node_emb": 64,
        "use_verbose": "False",
        "without_tanh": "False",
        "seed": 16,
    },
    "OPTC_base": {
        "epochs": 10,
        "iterations": 1,
        "dataset": "OPTC",
        "encoder": "FAUCON",
        "lr": 0.001,
        "alpha": 5,
        "beta": 30,
        "use_contra_loss": "False",
        "use_recon_loss": "True",
        "use_skip_connection": "True",
        "use_recurrent": "False",
        "keep_only_features": "all",
        "hid_node_emb": 128,
        "out_node_emb": 128,
        "use_verbose": "False",
        "without_tanh": "False",
        "seed": 42,
    },
}

BEST_CONFIGS = {
    # LANL configs
    "LANL_inductive_exp0": {
        **CONFIGS["LANL_base"],
        "dataset_name": "LANL_exp0",
    },
    "LANL_inductive_exp1": {
        **CONFIGS["LANL_base"],
        "dataset_name": "LANL_exp1",
    },
    "LANL_inductive_exp2": {
        **CONFIGS["LANL_base"],
        "dataset_name": "LANL_exp2",
    },
    "LANL_inductive_exp3": {
        **CONFIGS["LANL_base"],
        "dataset_name": "LANL_exp3",
    },
    "LANL_custom": {
        **CONFIGS["LANL_base"],
    },
    # OPTC configs
    "OPTC_inductive_exp0": {
        **CONFIGS["OPTC_base"],
        "dataset_name": "OPTC_exp0",
    },
    "OPTC_inductive_exp1": {
        **CONFIGS["OPTC_base"],
        "dataset_name": "OPTC_exp1",
    },
    "OPTC_inductive_exp2": {
        **CONFIGS["OPTC_base"],
        "dataset_name": "OPTC_exp2",
    },
    "OPTC_inductive_exp3": {
        **CONFIGS["OPTC_base"],
        "dataset_name": "OPTC_exp3",
    },
    "OPTC_custom": {
        **CONFIGS["OPTC_base"],
    },
}

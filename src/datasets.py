import argparse
import random
import sys
from pathlib import Path
from pprint import pprint
import numpy as np
import torch
import random

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.config import *
from utils.loaders import LOADERS
from utils.torch_device import COMPUTE_DEVICE as DEVICE
from utils.utils import preprocess_args

config = {
    "dataset": "LANL",
    "dataset_name": "",
    "use_no_self_loops": "True",
    "keep_only_features": "all",
    "node_features": "one_hot",
    "inductive_experiment": "None",
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=config["dataset"])
parser.add_argument(
    "--dataset_name", type=str, default=config["dataset_name"]
)

parser.add_argument(
    "--use_no_self_loops", type=str, default=config["use_no_self_loops"]
)
parser.add_argument(
    "--keep_only_features", type=str, default=config["keep_only_features"]
)
parser.add_argument(
    "--node_features", type=str, default=config["node_features"]
)
parser.add_argument(
    "--inductive_experiment", type=str, default=config["inductive_experiment"]
)

local_args = parser.parse_known_args()[0]
pprint(local_args.__dict__)

local_args = preprocess_args(local_args, config)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_loaders(args):
    train_loader, valid_loader, test_loader = loaders_constructor(args, do_compile=False)

    return train_loader, valid_loader, test_loader


def loaders_constructor(args, do_compile=True):
    conf = CONF(args)
    loader = LOADERS[args.dataset]

    train_loader = loader(
        ranges=conf["TIME_RANGES"]["TRAIN"],
        compiled_path=conf["TRAIN_PATH"],
        device=DEVICE,
        do_compile=do_compile,
        use_no_self_loops=args.use_no_self_loops,
        keep_only_features=args.keep_only_features,
        node_features=args.node_features,
        inductive_experiment=args.inductive_experiment,
        dataset_name=args.dataset_name,
    )
    valid_loader = loader(
        ranges=conf["TIME_RANGES"]["VALID"],
        compiled_path=conf["VALID_PATH"],
        device=DEVICE,
        do_compile=do_compile,
        use_no_self_loops=args.use_no_self_loops,
        keep_only_features=args.keep_only_features,
        node_features=args.node_features,
        dataset_name=args.dataset_name,
    )
    test_loader = loader(
        ranges=conf["TIME_RANGES"]["TEST"],
        compiled_path=conf["TEST_PATH"],
        device=DEVICE,
        do_compile=do_compile,
        use_no_self_loops=args.use_no_self_loops,
        keep_only_features=args.keep_only_features,
        node_features=args.node_features,
        dataset_name=args.dataset_name,
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    train_loader, valid_loader, test_loader = loaders_constructor(
        args=local_args,
        do_compile=True,
    )

    # Preprocessed files (one per minute) (.csv) ==> to compiled files (one per snapshot) (.pkl)
    print("Compiling dataset...")
    for i, data in enumerate(train_loader.iterate_lazily()):
        pass
    for i, data in enumerate(valid_loader.iterate_lazily()):
        pass
    for i, data in enumerate(test_loader.iterate_lazily()):
        pass

    print("Dataset compilation done successfully.")

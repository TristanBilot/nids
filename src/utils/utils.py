def preprocess_args(args, config):
    updates = {}
    for k, v in args.__dict__.items():
        config[k] = v  # To update with parser args
        if k.startswith("use"):  # To convert True/False strings to bools
            updates[k] = True if v == "True" else False
    args.__dict__.update(updates)
    return args


def replace_with_best_config(args, default_config, best_configs):
    if not args.config:
        return args

    # In order to consider other arguments except the best params, we memoize the args passed
    # in the CLI, to then replace the args in the best params.
    # NOTE: this only works when the arg in CLI is different from the default arg, I didn't find
    # a proper way to handle all cases.
    priority_args = {
        arg: val for arg, val in args.__dict__.items() if val != default_config[arg]
    }
    config_key = args.config

    if config_key in best_configs:
        args.__dict__.update(best_configs[config_key])
        args.__dict__.update(priority_args)
        return args

    raise ValueError(f"Please set the arg `config` to one of {best_configs.keys()}")

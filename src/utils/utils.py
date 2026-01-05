
import inspect
import json
import os
from typing import Optional
import random

import torch
from src import sam_optim

def initialize(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class LoadingBar:
    def __init__(self, length: int = 40):
        self.length = length
        self.symbols = ['┈', '░', '▒', '▓']

    def __call__(self, progress: float) -> str:
        p = int(progress * self.length*4 + 0.5)
        d, r = p // 4, p % 4
        return '┠┈' + d * '█' + ((self.symbols[r]) + max(0, self.length-1-d) * '┈' if p < self.length*4 else '') + "┈┨"


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class StepLR:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, epoch):
        if epoch < self.total_epochs * 3/10:
            lr = self.base
        elif epoch < self.total_epochs * 6/10:
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 8/10:
            lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 3

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

def resolve_sam_variant(name: str):
    try:
        return sam_optim[name], name
    except KeyError:
        pass
    lname = name.lower()

    keys = []
    if hasattr(sam_optim, "_variants"):
        keys = list(sam_optim._variants.keys())
    elif hasattr(sam_optim, "keys"):
        keys = list(sam_optim.keys())

    for k in keys:
        if str(k).lower() == lname:
            return sam_optim[k], k

    raise KeyError(f"Unknown sam_type: {name}. Available: {keys}")


def is_standard_sam_type(sam_type: Optional[str]) -> bool:
    if sam_type is None:
        return True
    s = sam_type.strip().lower()
    return s in {"", "none", "standard", "vanilla"}

def collect_method_aware_args(args, dataset, sam_variant_cls=None, sam_key=None, include_derived=True):
    args_dict = vars(args)

    common_keys = {
        "seed",
        "dataset", "arch_type", "dropout",
        "epochs", "batch_size", "label_smoothing",
        "optimizer", "lr", "weight_decay", "warmup_epochs",
        "sam_type", "adaptive",
    }

    if args_dict.get("optimizer", "").lower() == "sgd":
        common_keys.add("momentum")

    keys = set(common_keys)

    if sam_variant_cls is not None:
        sig = inspect.signature(sam_variant_cls.__init__)
        for p in sig.parameters.values():
            if p.name in {"self", "params", "base_optimizer", "model", "defaults"}:
                continue
            if p.name in args_dict:
                keys.add(p.name)

        if sam_key is not None:
            keys.add("_sam_key_canonical")

    filtered = {k: args_dict[k] for k in keys if k in args_dict}
    if sam_key is not None:
        filtered["_sam_key_canonical"] = sam_key

    if include_derived and isinstance(args_dict.get("sam_type", ""), str):
        if "bayesian" in args_dict["sam_type"].lower():
            try:
                Ndata = len(dataset.train.dataset)
                filtered["Ndata"] = Ndata
                filtered["prior_weight_decay"] = args.delta / Ndata
            except Exception:
                pass

    return filtered


def save_method_aware_args(args, dataset, save_dir: str, include_derived=True):
    sam_variant_cls = None
    sam_key = None

    if not is_standard_sam_type(args.sam_type):
        sam_variant_cls, sam_key = resolve_sam_variant(args.sam_type)

    to_save = collect_method_aware_args(
        args=args,
        dataset=dataset,
        sam_variant_cls=sam_variant_cls,
        sam_key=sam_key,
        include_derived=include_derived,
    )

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{args.dataset}_training_arguments.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_save, fh, indent=2, sort_keys=True, default=str)

    return path
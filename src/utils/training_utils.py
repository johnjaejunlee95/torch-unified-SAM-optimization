import inspect
import math 
import torch
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from src.data.cifar import CIFAR
from src.data.tinyImageNet import TinyImageNet
from .scheduler import LinearScheduler
from .utils import *
from src.model import MODEL_MAP

def get_train_dataset_len(dataloader) -> int:
    #  effort
    if hasattr(dataloader, "dataset"):
        try:
            return len(dataloader.dataset)
        except TypeError:
            pass
    return 0


def build_dataset(args):
    dataset_map = {
        "cifar10": (CIFAR, 10),
        "cifar100": (CIFAR, 100),
        "tinyimagenet": (TinyImageNet, 200),
    }
    dataset_class, num_classes = dataset_map[args.dataset]

    if "cifar" in args.dataset:
        dataset = dataset_class(args.batch_size, args.num_workers, num_classes)
    else:
        dataset = dataset_class(args.batch_size, args.num_workers)

    return dataset, num_classes


def build_model(args, num_classes: int, device: torch.device):

    model_class = MODEL_MAP[args.arch_type]
    if "wide" in args.arch_type:
        model = model_class(num_classes=num_classes).to(device)
    elif "resnet" in args.arch_type:
        model = model_class(num_classes=num_classes, dropout_rate=args.dropout).to(device)
    elif args.arch_type == "pyramidnet":
        model = model_class(110, 270, num_classes).to(device)
        
    return model


def build_base_optimizer(args):
    optimizer_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.optimizer.lower() == "sgd":
        base_opt_cls = torch.optim.SGD
        optimizer_kwargs["momentum"] = args.momentum
    else:
        base_opt_cls = torch.optim.Adam
    return base_opt_cls, optimizer_kwargs


def build_optimizer_and_scheduler(args, model, dataset):
    base_opt_cls, optimizer_kwargs = build_base_optimizer(args)

    use_sam = not is_standard_sam_type(args.sam_type)
    sam_name = (args.sam_type or "standard").strip()
    sam_lower = sam_name.lower()

    train_len = get_train_dataset_len(dataset.train)
    if train_len <= 0 and hasattr(dataset, "train") and hasattr(dataset.train, "dataset"):
        train_len = len(dataset.train.dataset)

    scheduler = None

    if use_sam:
        sam_variant_cls, sam_key = resolve_sam_variant(sam_name)

        sig = inspect.signature(sam_variant_cls.__init__)
        variant_params = {}
        for p in sig.parameters.values():
            if p.name in {"self", "params", "base_optimizer", "model", "defaults"}:
                continue
            if hasattr(args, p.name):
                variant_params[p.name] = getattr(args, p.name)

        sam_kwargs = {**optimizer_kwargs, **variant_params}
        base_opt_cls_or_inst = base_opt_cls

        if "bayesian" in sam_lower:
            Ndata = len(dataset.train.dataset)
            prior_weight_decay = args.delta / Ndata
            sam_kwargs["wdecay"] = prior_weight_decay
            sam_kwargs["Ndata"] = Ndata
            sam_kwargs["bn_update_once"] = True

        elif sam_lower == "gsam":
            base_opt_instance = base_opt_cls(model.parameters(), **optimizer_kwargs)
            rho_scheduler = LinearScheduler(
                T_max=args.epochs * len(dataset.train),
                max_value=args.rho_max,
                min_value=args.rho_min,
                warmup_steps=100,
            )
            sam_kwargs["rho_scheduler"] = rho_scheduler
            base_opt_cls_or_inst = base_opt_instance
        else:
            base_opt_cls_or_inst = base_opt_cls

        optimizer = sam_variant_cls(
            model.parameters(),
            base_optimizer=base_opt_cls_or_inst,
            model=model,
            **sam_kwargs,
        )
        training_type = sam_key

    else:
        optimizer = base_opt_cls(model.parameters(), **optimizer_kwargs)
        training_type = "Standard"

    if "bayesian" not in sam_lower:
        opt_for_scheduler = optimizer
        if hasattr(optimizer, "base_optimizer") and optimizer.base_optimizer is not None:
            opt_for_scheduler = optimizer.base_optimizer

        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            opt_for_scheduler,
            num_warmup_steps=args.warmup_epochs,
            num_training_steps=args.epochs,
            min_lr=1e-6,
        )

    if args.adaptive:
        training_type = "Adaptive_" + training_type

    return optimizer, scheduler, training_type, use_sam, sam_lower


def bayesian_lrfactor(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    step_t = float(epoch - warmup_epochs) / float(max(1, (total_epochs + 1 - warmup_epochs)))
    return 0.5 * (1.0 + math.cos(math.pi * step_t))


def train_one_epoch(
    *,
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    epoch: int,
    args,
    use_sam: bool,
    sam_lower: str,
):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()

    last_lr = None
    last_rho_lr = None

    for batch in loader:
        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_sam:
            if "bayesian" in sam_lower:
                lrf = bayesian_lrfactor(epoch, args.warmup_epochs, args.epochs)
                logits, loss, lr = optimizer.step(x=inputs, y=targets, lrfactor=lrf)
                last_lr = float(lr) if lr is not None else last_lr

            elif sam_lower == "esam":
                loss, logits = optimizer.step(inputs, targets, loss_fn)

            else:
                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    logits_ = model(inputs)
                    loss_ = loss_fn(logits_, targets)
                    loss_.backward()
                    return logits_, loss_

                logits, loss = optimizer.step(closure)

            if sam_lower == "gsam" and hasattr(optimizer, "update_rho_t"):
                last_rho_lr = optimizer.update_rho_t()

        else:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss_scalar = float(loss.mean().item()) if hasattr(loss, "mean") else float(loss)
            acc_scalar = (torch.argmax(logits, 1) == targets).float().mean().item()

            train_loss.update(loss_scalar, inputs.size(0))
            train_acc.update(acc_scalar, inputs.size(0))

    return train_loss.avg, train_acc.avg, last_lr, last_rho_lr


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    eval_loss = AverageMeter()
    eval_acc = AverageMeter()

    for batch in loader:
        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = loss_fn(logits, targets)

        loss_scalar = float(loss.mean().item()) if hasattr(loss, "mean") else float(loss)
        acc_scalar = (torch.argmax(logits, 1) == targets).float().mean().item()

        eval_loss.update(loss_scalar, inputs.size(0))
        eval_acc.update(acc_scalar, inputs.size(0))

    return eval_loss.avg, eval_acc.avg
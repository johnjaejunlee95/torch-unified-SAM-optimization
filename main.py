import argparse
import os
import json
import logging

import torch
import torch.nn as nn

from src.utils.log import setup_logger
from src.utils.utils import initialize
from src.utils.training_utils import *


def main(args):
    initialize(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, num_classes = build_dataset(args)
    model = build_model(args, num_classes, device)

    optimizer, scheduler, training_type, use_sam, sam_lower = build_optimizer_and_scheduler(args, model, dataset)

    sam_dir = args.sam_type if args.sam_type is not None else "standard"

    save_dir = f"src/save/{sam_dir}"
    saved_args_path = save_method_aware_args(args, dataset, save_dir=save_dir, include_derived=True)

    log_file_name = f"{sam_dir}_{args.arch_type}_{args.dataset}.log"
    os.makedirs(f"src/save/{sam_dir}/log/", exist_ok=True)

    logger = setup_logger(
        "Sharpness_Aware_Minimization",
        f"src/save/{sam_dir}/log/{log_file_name}",
        level=logging.INFO,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {total_params}")
    logger.info(f"Saved training args to: {saved_args_path}")
    logger.info(f"Starting training with {training_type}, on {args.dataset}, parameters: {total_params}.")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    for epoch in range(args.epochs):
        tr_loss, tr_acc, bayes_lr, _ = train_one_epoch(
            model=model,
            loader=dataset.train,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            args=args,
            use_sam=use_sam,
            sam_lower=sam_lower,
        )

        if scheduler is not None:
            scheduler.step()

        # val_loss, val_acc = evaluate(model, dataset.val, loss_fn, device)
        test_loss, test_acc = evaluate(model, dataset.test, loss_fn, device)

        if "bayesian" in sam_lower:
            lr = bayes_lr if bayes_lr is not None else args.lr
        elif scheduler is not None:
            lr = scheduler.get_last_lr()[0]
        else:
            lr = args.lr

        logger.info(
            f"Epoch: {epoch+1:03}/{args.epochs} | "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}% | "
            f"LR: {lr:.4e}"
        )
    
    model_save_path = f"src/save/{sam_dir}/state_dict/{training_type}_{args.arch_type}_{args.dataset}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Train image classification models with Sharpness-Aware Minimization (SAM) and its variants."
)

    # --- General ---
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of DataLoader worker processes.")

    # --- Dataset ---
    parser.add_argument("--dataset", default="cifar10", type=str, choices=["cifar10", "cifar100", "tinyimagenet"], help="Dataset to train/evaluate on.")

    # --- Model ---
    parser.add_argument("--arch_type", default="resnet18", type=str, help="Model architecture name (e.g., resnet18, wideresnet28, pyramidnet).")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout probability (if supported by the selected architecture).")

    # --- Training ---
    parser.add_argument("--epochs", default=200, type=int, help="Total number of training epochs.")
    parser.add_argument("--batch_size", default=128, type=int, help="Mini-batch size.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing factor used in CrossEntropyLoss (0.0 disables label smoothing).")

    # --- Optimizer ---
    parser.add_argument("--optimizer", default="sgd", type=str, choices=["adam", "sgd"], help="Base optimizer to use.")
    parser.add_argument("--lr", default=1e-2, type=float, help="Learning rate for the base optimizer.")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD (ignored for Adam).")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay (L2 regularization) for the base optimizer.")
    parser.add_argument("--warmup_epochs", default=5, type=int, help="Number of warmup epochs for the learning-rate schedule.")

    # --- SAM ---
    # NOTE: Set --sam_type to 'none' (or 'standard' depending on your implementation) to disable SAM.
    parser.add_argument("--sam_type", default="sam", type=str, help="SAM variant to use (e.g., SAM, ESAM, FisherSAM, FriendlySAM, BayesianSAM, GSAM, LookSAM). Use 'none' (or 'standard', depending on implementation) to disable SAM.")
    parser.add_argument("--beta", default=0.5, type=float, help="ESAM/related: mixing coefficient for efficient sharpness-aware updates (method-specific).")
    parser.add_argument("--adaptive", action="store_true", help="Enable ASAM-style adaptive perturbation scaling.")
    parser.add_argument("--rho", default=0.01, type=float, help="Perturbation radius (SAM neighborhood size).")
    parser.add_argument("--delta", default=40, type=float, help="Method-specific hyperparameter (used by certain SAM variants, e.g., BayesianSAM in this repo).")
    parser.add_argument("--msharpness", default=8, type=int, help="Method-specific hyperparameter controlling sharpness estimation granularity (e.g., BayesianSAM).")
    parser.add_argument("--gamma", default=0.1, type=float, help="Method-specific scaling/decay factor used by certain variants (e.g., BayesianSAM / ESAM depending on implementation).")
    parser.add_argument("--k", default=5, type=int, help="LookSAM: number of steps / look-ahead layers (method-specific).")
    parser.add_argument("--eta", default=0.2, type=float, help="FisherSAM: scaling factor for Fisher-based preconditioning (method-specific).")
    parser.add_argument("--gsam_alpha", default=0.01, type=float, help="GSAM: coefficient for gradient decomposition / alignment term (method-specific).")
    parser.add_argument("--alpha", default=1.0, type=float, help="LookSAM (or other variants): method-specific weighting coefficient.")
    parser.add_argument("--rho_max", default=2.0, type=float, help="GSAM: maximum perturbation radius for rho scheduling.")
    parser.add_argument("--rho_min", default=2.0, type=float, help="GSAM: minimum perturbation radius for rho scheduling.")
    parser.add_argument("--sigma", default=1.0, type=float, help="FriendlySAM: smoothing/noise scale (method-specific).")
    parser.add_argument("--lmbda", default=0.9, type=float, help="FriendlySAM: EMA/decay-style coefficient (method-specific).")
    args = parser.parse_args()

    sam_dir = args.sam_type if args.sam_type is not None else "standard"
    os.makedirs(f"src/save/{sam_dir}", exist_ok=True)
    args_path = f"src/save/{sam_dir}/arguments.json"
    with open(args_path, "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, sort_keys=True, default=str)

    main(args)

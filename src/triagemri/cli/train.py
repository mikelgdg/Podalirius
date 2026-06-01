#!/usr/bin/env python3
"""Train the Triage-MRI model.

Usage::

    triage-train --config_dir configs --output_dir outputs/run_001
    triage-train --resume_from outputs/run_001/checkpoints/last.ckpt
    triage-train --multi_sequence --anatomies brain prostate
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

_project_root = Path(__file__).resolve().parents[3]
_src = _project_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from triagemri.config import Config, load_config
from triagemri.data.datasets import create_dataloaders
from triagemri.models.triage import build_triage_model
from triagemri.training.losses import get_loss_fn
from triagemri.training.trainer import TriageLightningModule

logger = logging.getLogger("triagemri.train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Triage-MRI model.")
    parser.add_argument(
        "--config_dir",
        default="configs",
        help="Directory containing data/model/train YAML files.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/default",
        help="Root output directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Target device (cuda / cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed from the config.",
    )
    parser.add_argument(
        "--anatomies",
        nargs="+",
        default=None,
        choices=["brain", "prostate", "breast"],
        help="Which anatomies to train on (default: brain).",
    )
    parser.add_argument(
        "--multi_sequence",
        action="store_true",
        default=None,
        help="Enable multi-sequence input mode.",
    )
    parser.add_argument(
        "--logger",
        choices=["tensorboard", "wandb"],
        default="tensorboard",
        help="Logger backend (default: tensorboard).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_dataloaders(dataloaders: Dict[str, torch.utils.data.DataLoader]) -> None:
    n_train = len(dataloaders.get("train", []))
    n_val = len(dataloaders.get("val", []))
    n_test = len(dataloaders.get("test", []))
    if n_train == 0:
        raise RuntimeError(
            "Training DataLoader is empty.  "
            "Verify that data paths in configs/data.yaml are correct."
        )
    logger.info(
        "Dataset sizes — train: %d batches, val: %d, test: %d",
        n_train,
        n_val,
        n_test,
    )
    if n_val == 0:
        logger.warning(
            "Validation DataLoader is empty.  Early stopping and checkpointing "
            "will have no validation signal."
        )


def _validate_config(config: Config) -> None:
    required = ["training", "model", "data"]
    for key in required:
        if not hasattr(config, key):
            raise ValueError(
                f"Merged config is missing required section '{key}'.  "
                f"Ensure {key}.yaml exists in the config directory."
            )
    training = config.training
    for attr in ("max_epochs", "optimizer", "scheduler", "loss"):
        if not hasattr(training, attr):
            raise ValueError(f"config.training is missing required key '{attr}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("nibabel").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger.info("Triage-MRI training started")
    logger.info("  config dir : %s", args.config_dir)
    logger.info("  output dir : %s", args.output_dir)
    logger.info("  device     : %s", args.device)

    try:
        config = load_config(args.config_dir)
    except FileNotFoundError as exc:
        logger.error("Config error: %s", exc)
        sys.exit(1)
    _validate_config(config)

    if args.multi_sequence is not None:
        try:
            config.data.multi_sequence.enabled = args.multi_sequence
        except Exception:
            logger.warning("Could not set multi_sequence in config; using YAML default.")

    seed: int = (
        args.seed if args.seed is not None else int(config.training.get("seed", 42))
    )
    pl.seed_everything(seed, workers=True)
    logger.info("  seed       : %d", seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building model...")
    try:
        model = build_triage_model(config.model, device=args.device)
    except Exception:
        logger.error("Failed to build model:\n%s", traceback.format_exc())
        sys.exit(1)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    encoder_frozen = not any(
        p.requires_grad for p in model.encoder.parameters()
    )

    logger.info("  total params     : %s", f"{total_params:,}")
    logger.info("  trainable params : %s", f"{trainable_params:,}")
    logger.info("  encoder frozen   : %s", encoder_frozen)

    if args.resume_from is not None and args.anatomies is not None:
        try:
            ckpt = torch.load(args.resume_from, map_location="cpu")
            ckpt_anatomies = None
            if "hyper_parameters" in ckpt:
                hp = ckpt["hyper_parameters"]
                ckpt_anatomies = hp.get("anatomies") or hp.get("anatomy_mode")
            elif "config" in ckpt:
                cfg = ckpt["config"]
                if isinstance(cfg, dict):
                    ckpt_anatomies = cfg.get("anatomies") or cfg.get("anatomy_mode")
            if ckpt_anatomies is not None and set(ckpt_anatomies) != set(
                args.anatomies
            ):
                logger.warning(
                    "Resuming from checkpoint trained with anatomies %s "
                    "but --anatomies %s was specified. The model may behave "
                    "unexpectedly with the new anatomies.",
                    ckpt_anatomies,
                    args.anatomies,
                )
        except Exception:
            pass

    logger.info("Creating DataLoaders...")
    try:
        dataloaders = create_dataloaders(config, enabled_anatomies=args.anatomies)
    except Exception:
        logger.error("Failed to create DataLoaders:\n%s", traceback.format_exc())
        sys.exit(1)
    _validate_dataloaders(dataloaders)

    loss_cfg = config.training.loss
    loss_kwargs = loss_cfg.to_dict() if hasattr(loss_cfg, "to_dict") else dict(loss_cfg)
    loss_name: str = loss_kwargs.pop("name", "weighted_bce")
    try:
        loss_fn = get_loss_fn(loss_name, **loss_kwargs)
    except ValueError as exc:
        logger.error("Loss error: %s", exc)
        sys.exit(1)
    logger.info("  loss             : %s (kwargs=%s)", loss_name, loss_kwargs)

    logger.info("Building Lightning module...")
    lightning_module = TriageLightningModule(model, loss_fn, config)

    ckpt_cfg = config.training.checkpoint
    es_cfg = config.training.early_stopping

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="epoch_{epoch:02d}-auc_{val/roc_auc:.3f}",
        monitor=ckpt_cfg.monitor,
        mode=ckpt_cfg.mode,
        save_top_k=int(ckpt_cfg.save_top_k),
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor=es_cfg.monitor,
        patience=int(es_cfg.patience),
        min_delta=float(es_cfg.get("min_delta", 0.001)),
        mode=es_cfg.mode,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    if args.logger == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        pl_logger = WandbLogger(save_dir=str(output_dir), name="logs")
    else:
        pl_logger = TensorBoardLogger(save_dir=str(output_dir), name="logs")

    logger.info("Instantiating PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=int(config.training.max_epochs),
        accelerator=str(config.training.accelerator),
        devices=int(config.training.devices),
        precision=str(config.training.precision),
        accumulate_grad_batches=int(config.training.get("gradient_accumulation_steps", 1)),
        deterministic=bool(config.training.get("deterministic", False)),
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=pl_logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )

    logger.info("Starting training...")
    try:
        trainer.fit(
            lightning_module,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["val"],
            ckpt_path=args.resume_from,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving checkpoint...")
        if trainer.current_epoch > 0:
            trainer.save_checkpoint(
                str(output_dir / "checkpoints" / "interrupted.ckpt")
            )
            logger.info("Checkpoint saved.")
        sys.exit(0)
    except Exception:
        logger.error("Training failed:\n%s", traceback.format_exc())
        sys.exit(1)

    best_path = checkpoint_callback.best_model_path
    best_score = checkpoint_callback.best_model_score

    logger.info("Training finished.")
    if best_path:
        logger.info("  best checkpoint : %s", best_path)
    if best_score is not None:
        logger.info("  best %s : %.4f", es_cfg.monitor, float(best_score))
    else:
        logger.warning(
            "No best model score recorded (validation may have been empty)."
        )


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import yaml
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
except ImportError as exc:
    print("The requirements are not installed")
    print(str(exc))
    raise SystemExit(1)

from models import UnfoldIR
from models.losses import self_supervised_loss, supervised_loss
from utils.dataset import LowOnlyDataset, PairedImageDataset, SyntheticImageDataset


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_threads(cfg):
    value = cfg.get("num_threads", "auto")
    if value == "auto":
        value = max(1, min(8, (torch.get_num_threads() or 1)))
    torch.set_num_threads(int(value))


def build_model(cfg):
    return UnfoldIR(
        stage_num=cfg.get("stage_num", 3),
        shared_weights=cfg.get("shared_weights", True),
        feature_channels=cfg.get("feature_channels", 24),
        exposure_strength=cfg.get("exposure_strength", 1.0),
        contrast_strength=cfg.get("contrast_strength", 1.0),
        saturation_strength=cfg.get("saturation_strength", 1.0),
    )


def choose_dataset(cfg, forced_mode=None):
    image_size = int(cfg.get("image_size", 256))
    low_dir = cfg.get("train_low_dir", "data/train_low")
    high_dir = cfg.get("train_high_dir", "data/train_high")
    paired = PairedImageDataset(low_dir, high_dir, image_size)
    low_only = LowOnlyDataset(low_dir, image_size)
    synthetic = SyntheticImageDataset(
        [cfg.get("input_dir", "data/input"), cfg.get("test_dir", "data/test")],
        image_size=image_size,
        samples_per_image=cfg.get("synthetic", {}).get("samples_per_image", 2),
    )
    if forced_mode == "paired":
        return paired, "paired"
    if forced_mode == "self":
        return low_only, "self"
    if forced_mode == "synthetic":
        return synthetic, "synthetic"
    if len(paired) > 0:
        return paired, "paired"
    if len(low_only) > 0:
        return low_only, "self"
    return synthetic, "synthetic"


def save_checkpoint(model, optimizer, path, epoch, loss_value, mode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss_value,
            "mode": mode,
        },
        path,
    )


def train(args):
    cfg = load_config(args.config)
    configure_threads(cfg)
    dataset, mode = choose_dataset(cfg, args.mode)
    if len(dataset) == 0:
        print("No training data found. For training, place images in data/train_low or data/input or data/test.")
        return

    print(f"train mode: {mode}")
    loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 1)), shuffle=True, num_workers=0)
    model = build_model(cfg).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 0.0002)))
    weights = cfg.get("loss_weights", {})
    target_exposure = cfg.get("self_supervised", {}).get("target_exposure", 0.58)
    epochs = int(args.epochs or cfg.get("epochs", 20))
    checkpoint_path = args.checkpoint or cfg.get("checkpoint_path", "checkpoints/best_cpu.pth")
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        running = 0.0
        steps = 0
        progress = tqdm(loader, desc=f"epoch {epoch}/{epochs}", ncols=90)
        for batch in progress:
            if mode in {"paired", "synthetic"}:
                low, high, _ = batch
                enhanced, reflectances, illuminations, _ = model(low)
                loss = supervised_loss(enhanced, high, low, reflectances, illuminations, weights)
            else:
                low, _ = batch
                enhanced, reflectances, illuminations, _ = model(low)
                loss = self_supervised_loss(enhanced, low, reflectances, illuminations, weights, target_exposure)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item())
            steps += 1
            progress.set_postfix(loss=f"{running / max(steps, 1):.4f}")
        epoch_loss = running / max(steps, 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model, optimizer, checkpoint_path, epoch, best_loss, mode)
            print(f"New checkpoint saved: {checkpoint_path} | loss={best_loss:.4f}")
    print("Training is done!")


def main():
    parser = argparse.ArgumentParser(description="Train UnfoldIR")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["paired", "self", "synthetic"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()


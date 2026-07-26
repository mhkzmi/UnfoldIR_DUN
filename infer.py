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
    import torch.nn.functional as F
except ImportError as exc:
    print("requirements are not installed!")
    print(str(exc))
    raise SystemExit(1)

from models import UnfoldIR
from utils.image_io import list_images, load_image, save_tensor_image
from utils.metrics import psnr, ssim


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


def load_checkpoint_if_available(model, checkpoint):
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
        print(f"checkpoint loaded: {checkpoint}")
        return True
    print("Checkpoint not found; inference is performed with initial weights.")
    return False


@torch.no_grad()
def infer_resize(model, image, max_side=None):
    original_hw = image.shape[-2:]
    if max_side and max(original_hw) > max_side:
        scale = max_side / float(max(original_hw))
        new_hw = (max(1, int(original_hw[0] * scale)), max(1, int(original_hw[1] * scale)))
        work = F.interpolate(image, size=new_hw, mode="bilinear", align_corners=False)
    else:
        work = image
    enhanced, reflectances, illuminations, debug = model(work)
    if work.shape[-2:] != original_hw:
        enhanced = F.interpolate(enhanced, size=original_hw, mode="bilinear", align_corners=False)
        reflectances[-1] = F.interpolate(reflectances[-1], size=original_hw, mode="bilinear", align_corners=False)
        illuminations[-1] = F.interpolate(illuminations[-1], size=original_hw, mode="bilinear", align_corners=False)
    return enhanced, reflectances, illuminations, debug


@torch.no_grad()
def infer_tile(model, image, tile_size=384, overlap=48):
    _, _, h, w = image.shape
    if max(h, w) <= tile_size:
        return model(image)
    stride = max(32, tile_size - overlap)
    out = torch.zeros_like(image)
    weight = torch.zeros((1, 1, h, w), dtype=image.dtype)
    refl = torch.zeros_like(image)
    illum = torch.zeros((1, 1, h, w), dtype=image.dtype)
    stage_maps = None
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    if ys[-1] != max(0, h - tile_size):
        ys.append(max(0, h - tile_size))
    if xs[-1] != max(0, w - tile_size):
        xs.append(max(0, w - tile_size))
    for y in ys:
        for x in xs:
            patch = image[:, :, y : y + tile_size, x : x + tile_size]
            enhanced, reflectances, illuminations, debug = model(patch)
            ph, pw = patch.shape[-2:]
            out[:, :, y : y + ph, x : x + pw] += enhanced
            refl[:, :, y : y + ph, x : x + pw] += reflectances[-1].clamp(0, 1)
            illum[:, :, y : y + ph, x : x + pw] += illuminations[-1]
            if stage_maps is None and debug.get("stage_enhanced"):
                stage_maps = [torch.zeros_like(image) for _ in debug["stage_enhanced"]]
            if stage_maps is not None:
                for idx, stage in enumerate(debug.get("stage_enhanced", [])):
                    stage_maps[idx][:, :, y : y + ph, x : x + pw] += stage
            weight[:, :, y : y + ph, x : x + pw] += 1.0
    weight = weight.clamp_min(1.0)
    enhanced = out / weight
    reflectances = [refl / weight]
    illuminations = [illum / weight]
    stages = [stage / weight for stage in stage_maps] if stage_maps is not None else []
    return enhanced, reflectances, illuminations, {"stage_enhanced": stages}


def save_outputs(name, output_dir, enhanced, reflectances, illuminations, debug, save_debug):
    stem = Path(name).stem
    output_dir = Path(output_dir)
    save_tensor_image(enhanced, output_dir / f"{stem}_enhanced.png")
    save_tensor_image(reflectances[-1].clamp(0, 1), output_dir / f"{stem}_reflectance.png")
    save_tensor_image(illuminations[-1], output_dir / f"{stem}_illumination.png")
    if save_debug:
        for idx, stage in enumerate(debug.get("stage_enhanced", []), start=1):
            save_tensor_image(stage, output_dir / f"{stem}_stage_{idx}.png")


def run_self_test(cfg):
    configure_threads(cfg)
    model = build_model(cfg).eval()
    image = torch.rand(1, 3, 64, 80).pow(2.4)
    enhanced, reflectances, illuminations, _ = model(image)
    assert enhanced.shape == image.shape
    assert reflectances[-1].shape == image.shape
    assert illuminations[-1].shape[2:] == image.shape[2:]
    print("self-test done successfully!")


def main():
    parser = argparse.ArgumentParser(description="UnfoldIR CPU inference")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default=None, help="image directory")
    parser.add_argument("--output", default=None, help="output directory")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", choices=["resize", "tile"], default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gt", default=None, help="Optional ground truth folder for metrics")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.self_test:
        run_self_test(cfg)
        return

    configure_threads(cfg)
    input_path = Path(args.input or cfg.get("input_dir", "data/input"))
    output_dir = Path(args.output or cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(input_path)
    if not images:
        print("No input images found. Place images in data/input or specify --input.")
        return

    model = build_model(cfg).eval()
    checkpoint = args.checkpoint or cfg.get("checkpoint_path", "checkpoints/best_cpu.pth")
    load_checkpoint_if_available(model, checkpoint)

    infer_cfg = cfg.get("inference", {})
    mode = args.mode or infer_cfg.get("mode", "tile")
    gt_dir = Path(args.gt) if args.gt else None

    for path in images:
        image = load_image(path)
        if mode == "tile":
            enhanced, reflectances, illuminations, debug = infer_tile(
                model,
                image,
                tile_size=int(infer_cfg.get("tile_size", 384)),
                overlap=int(infer_cfg.get("tile_overlap", 48)),
            )
        else:
            enhanced, reflectances, illuminations, debug = infer_resize(
                model,
                image,
                max_side=int(infer_cfg.get("resize_max_side", 1024)),
            )
        save_outputs(path.name, output_dir, enhanced, reflectances, illuminations, debug, args.debug or cfg.get("save_stage_outputs", True))
        if gt_dir:
            gt_path = gt_dir / path.name
            if gt_path.exists():
                target = load_image(gt_path)
                if target.shape[-2:] != enhanced.shape[-2:]:
                    target = F.interpolate(target, size=enhanced.shape[-2:], mode="bilinear", align_corners=False)
                print(f"{path.name}: PSNR={psnr(enhanced, target):.2f}, SSIM={ssim(enhanced, target):.4f}")
        print(f"ذخیره شد: {output_dir / (path.stem + '_enhanced.png')}")


if __name__ == "__main__":
    main()

import os
import argparse
import torch
from torchvision.utils import save_image
from models_ct import ResGenerator128


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = ResGenerator128(
        z_dim=args.z_dim,
        base_channels=args.base_channels
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "generator" in checkpoint:
        generator.load_state_dict(checkpoint["generator"])
    else:
        generator.load_state_dict(checkpoint)

    generator.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    saved = 0
    with torch.no_grad():
        while saved < args.num_samples:
            current_batch = min(args.batch_size, args.num_samples - saved)

            z = torch.randn(current_batch, args.z_dim, device=device)
            fake = generator(z).cpu()

            for i in range(current_batch):
                save_image(
                    fake[i],
                    os.path.join(args.output_dir, f"fake_{saved + i:04d}.png"),
                    normalize=True,
                    value_range=(-1, 1)
                )

            saved += current_batch

            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"Saved {args.num_samples} images to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="generated")
    parser.add_argument("--num_samples", type=int, default=1604)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--base_channels", type=int, default=64)
    args = parser.parse_args()

    main(args)
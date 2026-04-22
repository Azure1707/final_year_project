import os
import argparse
import torch
from torch.utils.data import DataLoader

from datasets_ct import CTImageDataset
from models_ct import ResGenerator128, ResCritic128
from training_ct import train_wgan_gp


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CTImageDataset(
        root_dir=args.data_dir,
        image_size=args.image_size
    )
    print(f"Found {len(dataset)} images in: {args.data_dir}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    generator = ResGenerator128(z_dim=args.z_dim, base_channels=args.base_channels).to(device)
    critic = ResCritic128(base_channels=args.base_channels).to(device)

    train_wgan_gp(
        generator=generator,
        critic=critic,
        dataloader=dataloader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        z_dim=args.z_dim,
        lambda_gp=args.lambda_gp,
        critic_iters=args.critic_iters,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        save_every=args.save_every,
        sample_every=args.sample_every
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--image_size", type=int, default=128, choices=[128])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--critic_iters", type=int, default=3)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--sample_every", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
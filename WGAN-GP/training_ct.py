import os
import torch
import torch.optim as optim
from torchvision.utils import save_image


def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real + (1.0 - epsilon) * fake
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


def save_checkpoint(generator, critic, opt_gen, opt_critic, epoch, output_dir, filename=None):
    if filename is None:
        filename = f"checkpoint_{epoch}.pth"

    checkpoint_path = os.path.join(output_dir, filename)
    torch.save({
        "epoch": epoch,
        "generator": generator.state_dict(),
        "critic": critic.state_dict(),
        "opt_gen": opt_gen.state_dict(),
        "opt_critic": opt_critic.state_dict()
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def save_samples(generator, fixed_noise, epoch, output_dir):
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
        sample_path = os.path.join(output_dir, f"samples_epoch_{epoch}.png")
        save_image(fake, sample_path, nrow=4, normalize=True, value_range=(-1, 1))
    generator.train()
    print(f"Saved samples: {sample_path}")


def train_wgan_gp(
    generator,
    critic,
    dataloader,
    device,
    epochs=100,
    lr=1e-4,
    z_dim=128,
    lambda_gp=10.0,
    critic_iters=3,
    output_dir="outputs",
    checkpoint_path=None,
    save_every=5,
    sample_every=1
):
    os.makedirs(output_dir, exist_ok=True)

    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

    start_epoch = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint["generator"])
        critic.load_state_dict(checkpoint["critic"])
        opt_gen.load_state_dict(checkpoint["opt_gen"])
        opt_critic.load_state_dict(checkpoint["opt_critic"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from checkpoint: {checkpoint_path}")
        print(f"Starting from epoch: {start_epoch}")

    fixed_noise = torch.randn(16, z_dim, device=device)

    for epoch in range(start_epoch, epochs):
        for batch_idx, real in enumerate(dataloader):
            real = real.to(device)
            batch_size = real.size(0)

            for _ in range(critic_iters):
                z = torch.randn(batch_size, z_dim, device=device)
                fake = generator(z)

                critic_real = critic(real)
                critic_fake = critic(fake.detach())
                gp = gradient_penalty(critic, real, fake.detach(), device)

                loss_critic = -(critic_real.mean() - critic_fake.mean()) + lambda_gp * gp

                opt_critic.zero_grad(set_to_none=True)
                loss_critic.backward()
                opt_critic.step()

            z = torch.randn(batch_size, z_dim, device=device)
            fake = generator(z)
            loss_gen = -critic(fake).mean()

            opt_gen.zero_grad(set_to_none=True)
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss D: {loss_critic.item():.4f} "
                    f"Loss G: {loss_gen.item():.4f}"
                )

        if (epoch + 1) % sample_every == 0:
            save_samples(generator, fixed_noise, epoch + 1, output_dir)

        if (epoch + 1) % save_every == 0:
            save_checkpoint(generator, critic, opt_gen, opt_critic, epoch, output_dir)

    save_checkpoint(generator, critic, opt_gen, opt_critic, epochs - 1, output_dir, filename="checkpoint_final.pth")
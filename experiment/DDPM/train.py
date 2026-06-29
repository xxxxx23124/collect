from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from experiment.DDPM.DDPM import DDPM
from experiment.DDPM.Tools.ExpertMonitor import ExpertMonitor


def run_training(
        model_cls,
        dataset,
        epochs=200,
        batch_size=64,
        accumulation_steps=4,  # 累积几步更新一次
        lr=2e-4,
        device="cuda",
        timesteps=300,
        image_size=64,
        model_kwargs=None,
        save_path=None,
        save_model=False,
        checkpoint_dir=None,
        save_every_steps=None,
        sample_on_save=True,
        sample_dir=".",
        sample_every=1,
        num_samples=16,
        num_workers=4,
        use_monitor=False,
        max_train_steps=None,
):
    # 1. 初始化模型和 DDPM
    model_kwargs = model_kwargs or {}
    unet = model_cls(**model_kwargs)
    ddpm = DDPM(model=unet, timesteps=timesteps).to(device)

    # 2. 数据加载器
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )

    # 3. 优化器和调度器
    optimizer = optim.AdamW(ddpm.parameters(), lr=lr, weight_decay=0)
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else sample_dir / "checkpoints"
    if save_model or save_every_steps is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 按 optimizer step 调度，更适合大数据集和梯度累积训练。
    total_optimizer_steps = max(1, (len(dataloader) * epochs) // accumulation_steps)
    if max_train_steps is not None:
        total_optimizer_steps = min(total_optimizer_steps, max_train_steps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=1e-6)
    if use_monitor:
        monitor = ExpertMonitor(unet, log_dir=r"runs/ddpm_experiment")

    def save_checkpoint(epoch, global_step):
        if save_path is not None and save_every_steps is None:
            checkpoint_path = Path(save_path)
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step:06d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": ddpm.model.state_dict(),
                "ddpm_state_dict": ddpm.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "timesteps": timesteps,
                "image_size": image_size,
                "batch_size": batch_size,
                "accumulation_steps": accumulation_steps,
                "model_kwargs": model_kwargs,
            },
            checkpoint_path,
        )
        print(f"💾 Checkpoint saved to {checkpoint_path}")

    def save_samples(global_step):
        print("🎨 Sampling images...")
        generated_imgs = ddpm.sample(num_samples=num_samples, img_size=image_size)
        sample_path = sample_dir / f"sample_step_{global_step:06d}.png"
        save_image(generated_imgs, sample_path, nrow=max(1, min(4, num_samples)))
        print(f"✅ Saved sample to {sample_path}")

    # 4. 训练循环
    batch_loss = 0
    accum_counter = 0
    global_step = 0
    stop_training = False
    optimizer.zero_grad(set_to_none=True)

    print(f"🚀 Start Training on {device}...")
    print(f"   Batch Size: {batch_size}, Accumulation: {accumulation_steps}")
    print(f"   Effective Batch Size: {batch_size * accumulation_steps}")
    print(f"   DDPM Timesteps: {timesteps}, Image Size: {image_size}")
    print(f"   Planned Optimizer Steps: {total_optimizer_steps}")

    for epoch in range(epochs):
        ddpm.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(progress_bar):
            # 假设 dataset 返回的是 (image, label) 或者 image
            # 我们只需要 image
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device, non_blocking=use_cuda)

            # 重要：DDPM 假设输入在 [-1, 1]，如果 DataLoader 输出是 [0, 1]，需要转换
            # images = images * 2.0 - 1.0

            # ================= 1. 计算主 Loss (依赖数据) =================
            loss_main = ddpm.compute_loss(images)

            # 梯度累积
            # 除以累积步数，因为 backward 会累加梯度
            loss_main = loss_main / accumulation_steps
            # 反向传播主 Loss
            loss_main.backward()

            # 还原主 Loss 数值用于打印
            loss_val = loss_main.item() * accumulation_steps
            epoch_loss += loss_val

            # 用于日志记录 (还原数值)
            batch_loss += loss_main.item()

            # 计数器 +1
            accum_counter += 1

            # 只有满足累积步数时才进行更新
            if accum_counter % accumulation_steps == 0:
                # 正则化 Loss 只与权重有关，与 Batch 大小无关，
                # 所以不需要除以 accumulation_steps，直接加一次梯度即可。
                aux_loss = unet.get_auxiliary_loss()
                aux_loss.backward()

                # 梯度裁剪 (Max Norm 通常设为 1.0)
                torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if use_monitor:
                    # 此时 monitor.batch_buffer 里已经攒了 N 个 mini-batch 的数据
                    # 调用 log_and_reset 进行结算、记录并清空
                    monitor.log_and_reset(global_step)

                    # 顺便记录一下 Loss 和 LR
                    monitor.writer.add_scalar("Train/Loss", batch_loss, global_step)
                    monitor.writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], global_step)

                # 重置累积的 batch_loss
                batch_loss = 0

                if save_every_steps is not None and save_every_steps > 0 and global_step % save_every_steps == 0:
                    save_checkpoint(epoch + 1, global_step)
                    if sample_on_save:
                        save_samples(global_step)

                if max_train_steps is not None and global_step >= max_train_steps:
                    stop_training = True
                    break

            progress_bar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}", "global step:": f"{global_step}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"📉 Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        if save_model and save_every_steps is None:
            save_checkpoint(epoch + 1, global_step)

        if save_every_steps is None and sample_every > 0 and (epoch + 1) % sample_every == 0:
            save_samples(global_step)

        if stop_training:
            break

    if use_monitor:
        monitor.close()
    print("✅ Training Finished!")
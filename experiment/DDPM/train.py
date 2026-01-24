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
        save_path="ddpm_cat.pth",
        use_monitor=False,
):
    # 1. 初始化模型和 DDPM
    unet = model_cls()
    ddpm = DDPM(model=unet, timesteps=1000).to(device)

    # 2. 数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3. 优化器和调度器
    optimizer = optim.AdamW(ddpm.parameters(), lr=lr, weight_decay=0)

    # 余弦退火调度器，T_max 设为总步数或 Epoch 数
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    if use_monitor:
        monitor = ExpertMonitor(unet, log_dir=r"runs/ddpm_experiment")
    # 4. 训练循环
    batch_loss = 0
    accum_counter = 0
    global_step = 0

    print(f"🚀 Start Training on {device}...")
    print(f"   Batch Size: {batch_size}, Accumulation: {accumulation_steps}")
    print(f"   Effective Batch Size: {batch_size * accumulation_steps}")

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

            images = images.to(device)

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
            avg_loss = epoch_loss / len(dataloader)

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
                optimizer.zero_grad()
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

            progress_bar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}", "global step:": f"{global_step}"})

        # 每个 Epoch 结束后调整学习率
        scheduler.step()

        print(f"📉 Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        # 每隔 10 个 epoch 保存一次，并尝试采样看效果
        # if (epoch + 1) % 10 == 0:
            # torch.save(ddpm.model.state_dict(), save_path)
            # print(f"💾 Model saved to {save_path}")

        # 采样并保存图片
        print("🎨 Sampling images...")
        generated_imgs = ddpm.sample(num_samples=16, img_size=64)
        save_image(generated_imgs, f"output_epoch_{epoch + 1}.png", nrow=4)
        print(f"✅ Saved sample to output_epoch_{epoch + 1}.png")

    if use_monitor:
        monitor.close()
    print("✅ Training Finished!")
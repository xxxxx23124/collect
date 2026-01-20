import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from experiment.DDPM.Model.UNet import TimeAwareCondConv2d

class ExpertMonitor:
    def __init__(self, model, log_dir="runs/ddpm_experts"):
        self.hooks = []
        # 用 list 来存储多次 forward 的结果
        self.batch_buffer = defaultdict(list)
        self.layer_names = []
        self.writer = SummaryWriter(log_dir)
        self._register_hooks(model)
        print(f"👀 累积式监控器已启动！日志将保存到: {log_dir}")

    def _register_hooks(self, model):
        """
        自动遍历模型，找到所有的 TimeAwareCondConv2d。
        """
        # 遍历所有模块，寻找目标层
        for name, module in model.named_modules():
            # 我们只关心 TimeAwareCondConv2d
            if isinstance(module, TimeAwareCondConv2d):
                # 注册 Hook
                hook = module.router.register_forward_hook(
                    self._get_hook_fn(name)
                )
                self.hooks.append(hook)
                self.layer_names.append(name)
        print(f"共监控了 {len(self.hooks)} 个 TimeAwareCondConv2d 层")

    def _get_hook_fn(self, layer_name):
        """
        生成闭包钩子函数，为了记住是哪一层的名字。
        """

        def hook(module, input, output):
            # output: (MiniBatch, num_experts) -> Logits
            # 如果模型不在训练模式，直接无视
            if not module.training:
                return
            with torch.no_grad():
                probs = F.softmax(output, dim=1)
                # 计算当前 Mini-Batch 的平均使用率
                # 这里得到 [num_experts] 的向量
                avg_usage = probs.mean(dim=0).detach().cpu()

                # Append 到缓存列表中
                self.batch_buffer[layer_name].append(avg_usage)

        return hook

    def log_and_reset(self, global_step):
        """
        这个函数要在 optimizer.step() 之后调用。
        它会结算过去几次 forward 的总账，写入 TensorBoard，然后清空缓存。
        """
        for layer_name, usage_list in self.batch_buffer.items():
            if not usage_list:
                continue

            # usage_list 是一个列表，里面有 accumulation_steps 个 tensor
            # 比如 10 个 [4] 的 tensor
            # 我们将它们 stack 起来变成 [10, 4]，然后对 dim=0 求平均
            # 这样得到的就是整个 Effective Batch 的平均专家使用率
            accumulated_usage = torch.stack(usage_list).mean(dim=0)

            # 1. 记录每个专家的曲线
            for i, u in enumerate(accumulated_usage):
                self.writer.add_scalar(f"Expert_Usage/{layer_name}/Exp_{i}", u, global_step)

            # 2. 记录熵 (反映负载均衡度)
            entropy = -torch.sum(accumulated_usage * torch.log(accumulated_usage + 1e-9))
            self.writer.add_scalar(f"Expert_Entropy/{layer_name}", entropy, global_step)

        # 【关键】清空缓存，迎接下一个 Accumulation Cycle
        self.batch_buffer.clear()

    def close(self):
        for h in self.hooks:
            h.remove()
        self.writer.close()
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from experiment.DDPM.Model.UNet import TimeAwareCondConv2d

class ExpertMonitor:
    def __init__(self, model, log_dir="runs/ddpm_experts"):
        self.hooks = []
        self.expert_stats = defaultdict(list)
        self.layer_names = []
        # 初始化 TensorBoard Writer
        self.writer = SummaryWriter(log_dir)
        self._register_hooks(model)
        print(f"👀 监控器已启动！日志将保存到: {log_dir}")

    def _register_hooks(self, model):
        """
        自动遍历模型，找到所有的 TimeAwareCondConv2d，并狠狠地挂上钩子。
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
            # output: (B, num_experts) -> Logits
            with torch.no_grad():
                # 计算 Softmax 得到概率
                probs = F.softmax(output, dim=1)
                # 计算当前 Batch 的平均使用率
                avg_usage = probs.mean(dim=0).detach().cpu()
                self.expert_stats[layer_name] = avg_usage

        return hook

    def log_step(self, global_step):
        """
        将当前这一步的数据写入 TensorBoard
        """
        for layer_name, usage in self.expert_stats.items():
            # usage 是一个向量，例如 [0.25, 0.25, 0.25, 0.25]
            # 我们把它拆开记录，这样你可以看到每个专家的曲线

            # 记录每个专家的曲线
            for i, u in enumerate(usage):
                self.writer.add_scalar(f"Expert_Usage/{layer_name}/Exp_{i}", u, global_step)

            # 记录熵 (Entropy)
            # 熵越高(越接近最大值)，说明负载越均衡；熵越低，说明专家坍塌了
            # H = -sum(p * log(p))
            # 加上 1e-9 防止 log(0)
            entropy = -torch.sum(usage * torch.log(usage + 1e-9))
            self.writer.add_scalar(f"Expert_Entropy/{layer_name}", entropy, global_step)

    def print_summary_to_console(self, tqdm_bar=None):
        """
        如果你非要看控制台，用这个方法。
        它会使用 tqdm.write 避免打断进度条。
        """
        msg = "\n📊 [Expert Monitor Snapshot]\n"
        for name in self.layer_names:
            if name in self.expert_stats:
                u = self.expert_stats[name]
                # 格式化一下，比如 [0.25, 0.25, 0.25, 0.25]
                u_str = " | ".join([f"{x:.2f}" for x in u])
                msg += f"  {name[-20:]:<20}: [{u_str}]\n"

        if tqdm_bar:
            tqdm_bar.write(msg)
        else:
            print(msg)

    def close(self):
        for h in self.hooks:
            h.remove()
        self.writer.close()
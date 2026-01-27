"""训练主脚本占位"""
import os
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.config import get_config
# --- 导入我们之前实现的模块 ---
from data.IDRiD_dataset import IDRiDDataset
from data.transforms_IDRiD import Compose, Resize, HistogramEqualization, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, \
    RandomRotate
from models.rpn_net import RPN
from utils.losses import RPNLoss
from utils.metrics import MetricCalculator
from utils.visualization import VisualizationUtils
from utils.label_generator import LabelGenerator
from utils.logger import get_logger
save_dir = "./experiments/logs"
logger = get_logger(save_dir, run_name="RPN")
def get_args():
    parser = argparse.ArgumentParser(description='Train RPN for Fundus Lesion Segmentation')
    # 路径设置
    parser.add_argument('--data_root', type=str, required=True, help='Path to DDR dataset root')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')

    # 任务设置 (对应论文不同病灶使用不同配置)
    parser.add_argument('--lesion_type', type=str, default='MA', choices=['MA', 'HE', 'EX', 'SE'],
                        help='Lesion type to train (MA: Microaneurysms, etc.)')

    # 超参数 (对应论文 Table 1 & Section 4.3.2)
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs ')
    parser.add_argument('--batch_size', type=int, default=3, help='Train batch size ')
    parser.add_argument('--val_batch_size', type=int, default=2, help='Validation batch size ')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate [cite: 395]')
    parser.add_argument('--lr_milestones', nargs='+', type=int, default=[250, 350, 400],
                        help='LR decay milestones [cite: 395]')
    parser.add_argument('--input_size', nargs='+', type=int, default=[1000, 960],
                        help='Input size H W (DDR: 960x1000) [cite: 391]')

    # 模型配置 (对应论文 Table 4)
    # 默认值设为论文中针对 MA 的最佳配置
    parser.add_argument('--rsm_kernel', type=int, default=35, help='RSM generation kernel size')
    parser.add_argument('--pfm_kernel', type=int, default=17, help='PFM generation kernel size (MA=17)')
    parser.add_argument('--backbone_layers', type=int, default=3, help='Backbone depth (MA=3, others=5/7) [cite: 442]')

    # 系统设置
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu/mps)')
    parser.add_argument('--vis_freq', type=int, default=20, help='Visualize results every N epochs')

    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, label_gen):
    model.train()
    running_loss = 0.0
    running_l_per = 0.0
    running_l_ctr = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", unit="img")

    for batch in pbar:
        # 1. 准备数据
        images = batch['image'].to(device)
        pfm_gt = batch['pfm_gt'].to(device)  # 0, 1, 2
        mask_binary = batch['mask_binary'].to(device)
        with torch.no_grad():
            rsm_gt = label_gen.generate_rsm(mask_binary)
        rsm_gt = torch.clamp(rsm_gt, 0.0, 1.0)
        # 2. 前向传播
        # rsm_preds 是列表 (对应多个 PVB 分支)
        # pfm_logits 是中央分支输出
        rsm_preds, pfm_logits = model(images)

        # 3. 计算 Loss
        loss, loss_dict = criterion(rsm_preds, rsm_gt, pfm_logits, pfm_gt)

        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 5. 记录
        running_loss += loss.item()
        running_l_per += loss_dict['loss_per']
        running_l_ctr += loss_dict['loss_ctr']

        pbar.set_postfix(loss=loss.item(), per=loss_dict['loss_per'], ctr=loss_dict['loss_ctr'])

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, metric_calc, device, epoch, save_vis_dir=None,label_gen=None):
    model.eval()
    metric_calc.reset()
    val_loss = 0.0

    # 为了可视化，只取第一个 Batch
    vis_batch = None

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Valid]", unit="img")
        for i, batch in enumerate(pbar):
            images = batch['image'].to(device)
            pfm_gt = batch['pfm_gt'].to(device)  # 用于计算 Validation Loss (带忽略区域)
            original_mask = batch['original_mask']  # 用于计算 Metrics (原始 0/1 GT)

            mask_binary = batch['mask_binary'].to(device)
            rsm_gt = label_gen.generate_rsm(mask_binary)
            # 前向传播 (此时 model.training=False，返回的是 pfm_prob)
            # 但我们需要 Loss，所以这里手动调用 model.forward 的逻辑或者暂时把 model 设为 train 模式获取中间值
            # 为了简单，我们在 model 推理模式下通常只拿结果。
            # 如果想算 Val Loss，建议临时改一下 model 的返回逻辑，或者分开处理。
            # 这里演示: 直接使用 rsm_preds 和 pfm_logits 来算 Loss

            # --- Hack: 临时开启 model.training 标志来获取中间输出用于算 Loss ---
            model.train()
            rsm_preds, pfm_logits = model(images)
            model.eval()
            # -------------------------------------------------------------

            # 计算 Val Loss
            loss, _ = criterion(rsm_preds, rsm_gt, pfm_logits, pfm_gt)
            val_loss += loss.item()

            # 计算 Metrics (需将 logits 转为 prob)
            pfm_prob = torch.sigmoid(pfm_logits)
            metric_calc.update(pfm_prob, original_mask)

            # 保存第一个 Batch 用于可视化
            if i == 0 and save_vis_dir:
                # 取最后一个 RSM 分支作为可视化对象
                last_rsm_pred = rsm_preds[-1] if isinstance(rsm_preds, list) else rsm_preds
                vis_batch = (images, rsm_gt, last_rsm_pred, pfm_gt, pfm_prob)

    # 汇总指标
    metrics = metric_calc.compute()
    avg_loss = val_loss / len(loader)

    print(f"\nVal Loss: {avg_loss:.4f} | AUC-PR: {metrics['auc_pr_list']:.4f} | Dice: {metrics['dice_list']:.4f}")

    # 可视化保存
    if save_vis_dir and vis_batch:
        VisualizationUtils.plot_batch_results(
            *vis_batch,
            save_path=os.path.join(save_vis_dir, f"epoch_{epoch}_vis.png")
        )

    return metrics['auc_pr_list'], metrics


def main():
    # 1. 命令行参数解析 (只保留最核心的路径参数和覆盖选项)
    parser = argparse.ArgumentParser(description='Train RPN with Config')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file (e.g., configs/config_ma.yaml)')
    parser.add_argument('--data_root', type=str, required=True, help='Path to DDR dataset root')

    # 允许从命令行覆盖 YAML 中的关键参数 (可选)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)

    args = parser.parse_args()

    # 2. 加载配置 (YAML + Args 融合)
    # cfg 对象现在包含了 yaml 里的所有参数，也可以通过 cfg.key 访问
    cfg = get_config(args)

    logger.info(f"====== RPN Training Configuration ======")
    logger.info(f"Task: {cfg.lesion_type}")
    logger.info(f"Config File: {args.config}")
    logger.info(f"RSM Kernel: {cfg.rsm_kernel} | PFM Kernel: {cfg.pfm_kernel}")
    logger.info(f"Backbone Layers: {cfg.backbone_layers} | PVB Layers: {cfg.pvb_layer_indices}")
    logger.info(f"========================================")

    # 3. 准备目录
    # 拼接路径: ./checkpoints/MA
    final_save_dir = os.path.join(cfg.save_dir, cfg.lesion_type)
    vis_dir = os.path.join(final_save_dir, 'vis')

    os.makedirs(final_save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 设置设备
    device_name = cfg.device if cfg.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    logger.info(f"Using device: {device}")

    # 4. 准备数据转换 (Augmentation)
    # Train: 尺寸调整 + 翻转旋转 + 均衡化
    train_transforms = Compose([
        Resize((cfg.input_size[0], cfg.input_size[1])),  # YAML中是 [1000, 960]
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotate(15),
        HistogramEqualization(clahe=True),
        ToTensor()
    ])

    # Valid: 只有 Resize + 均衡化 + ToTensor
    val_transforms = Compose([
        Resize((cfg.input_size[0], cfg.input_size[1])),
        HistogramEqualization(clahe=True),
        ToTensor()
    ])

    # 5. 数据集与加载器
    # 注意: 参数全部从 cfg 中获取
    train_set = IDRiDDataset(
        root_dir=cfg.data_root,
        mode='train',
        lesion_type=cfg.lesion_type,
        transforms=train_transforms,
        rsm_kernel_size=cfg.rsm_kernel,
        pfm_kernel_size=cfg.pfm_kernel
    )
    val_set = IDRiDDataset(
        root_dir=cfg.data_root,
        mode='valid',
        lesion_type=cfg.lesion_type,
        transforms=val_transforms,
        rsm_kernel_size=cfg.rsm_kernel,
        pfm_kernel_size=cfg.pfm_kernel
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    logger.info(f"Dataset Loaded. Train: {len(train_set)}, Val: {len(val_set)}")

    # 6. 初始化模型
    # 核心修改: 直接使用配置文件中的 pvb_layer_indices 和 cvb_fusion_layers
    # 这样就可以通过 YAML 控制网络结构 (例如 MA 用 f_d2, HE 用 f_d3+f_d4)
    model = RPN(
        num_classes=1,
        backbone_base_channels=64,
        pvb_layer_indices=cfg.pvb_layer_indices,  # 从配置读取
        cvb_fusion_layers=cfg.cvb_fusion_layers  # 从配置读取
    ).to(device)

    # --- 新增: 初始化 LabelGenerator 并移至 GPU ---
    label_gen = LabelGenerator(
        rsm_kernel_size=cfg.rsm_kernel,
        pfm_kernel_size=cfg.pfm_kernel
    ).to(device)  # <--- 关键！放到显存里

    # 7. 损失函数与优化器
    criterion = RPNLoss()
    metric_calc = MetricCalculator()

    # 优化器配置
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.base_lr,
        weight_decay=cfg.get('weight_decay', 0)  # 使用 get 防止旧配置没有该项报错
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.lr_milestones,
        gamma=0.1
    )

    # 8. 训练循环
    best_auc_pr = 0.0

    for epoch in range(1, cfg.epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch,label_gen)

        # Scheduler Step
        scheduler.step()

        # Validation
        # 根据 vis_freq 决定是否保存图片
        should_visualize = (epoch % cfg.vis_freq == 0)
        save_vis_path = vis_dir if should_visualize else None

        auc_pr, metrics = validate(
            model, val_loader, criterion, metric_calc, device, epoch,
            save_vis_dir=save_vis_path,label_gen=label_gen
        )

        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch}/{cfg.epochs} | Time: {epoch_time:.0f}s | Train Loss: {train_loss:.4f}")

        # 保存最佳模型
        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            save_path = os.path.join(final_save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            logger.info(f">>> New Best AUC-PR: {best_auc_pr:.4f} saved to {save_path}")

        # 定期保存 checkpoint (例如每 50 轮)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(final_save_dir, f'epoch_{epoch}.pth'))

    logger.info(f"Training Complete. Best AUC-PR: {best_auc_pr:.4f}")

if __name__ == "__main__":
    main()
RPN1_0/
├── configs/                        # 配置文件 (对应论文 Table 1 & Table 4)
│   ├── config_ma.yaml              # 微动脉瘤(MA)专用配置 (Kernel=17, Layers=3)
│   ├── config_ex.yaml              # 硬渗出(EX)专用配置
│   ├── config_he.yaml              # 出血(HE)专用配置
│   └── config_se.yaml              # 软渗出(SE)专用配置
│
├── data/                           # 数据加载与预处理
│   ├── __init__.py
│   ├── ddr_dataset.py              # 自定义 Dataset 类 (实现 DDR 读取)
│   └── transforms.py               # 图像增强: 翻转、旋转、直方图均衡化 [cite: 388]
│
├── models/                         # 模型架构
│   ├── __init__.py
│   ├── blocks.py                   # 基础组件: SEBlock, DoubleConv
│   ├── backbone.py                 # 主干网络 (我们刚才实现的 U-Net Encoder-Decoder)
│   ├── peripheral.py               # 周边视觉分支 (PVB)
│   ├── central.py                  # 中央视觉分支 (CVB)
│   └── rpn_net.py                  # 完整的 RPN 模型组装
│
├── utils/                          # 工具类
│   ├── __init__.py
│   ├── label_generator.py          # 核心: 将 GT 转为 RSM 和 PFM (核心算法)
│   ├── losses.py                   # 损失函数: L_per + L_ctr (带 mask 的 BCE)
│   ├── metrics.py                  # 评价指标: AUC-PR, F1, IoU [cite: 373, 382]
│   └── visualization.py            # 可视化工具: 绘制 RSM 热力图和分割结果
│
├── train.py                        # 训练主脚本
├── test.py                         # 测试与推理脚本
├── requirements.txt                # 依赖库 (torch, opencv-python, scikit-learn等)
└── README.md                       # 项目说明
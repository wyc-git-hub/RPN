import logging
import os
import sys
from datetime import datetime


def get_logger(save_dir, run_name="train"):
    """
    初始化日志记录器
    Args:
        save_dir (str): 日志文件保存的文件夹路径
        run_name (str): 实验名称，用于文件名前缀
    Returns:
        logger: 配置好的日志对象
    """
    # 1. 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. 创建日志对象
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)  # 设置记录级别

    # 防止重复添加 Handler (如果在一个脚本里多次调用 get_logger)
    if logger.hasHandlers():
        return logger

    # 3. 定义格式: [时间] [级别] 消息
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 4. 创建文件处理器 (FileHandler): 写入文件
    # 文件名示例: logs/train_2023-10-27_14-30-00.log
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(save_dir, f'{run_name}_{time_str}.log')

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 5. 创建控制台处理器 (StreamHandler): 输出到屏幕
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 6. 打印一条初始信息
    logger.info(f"Log initialized. Saving logs to: {log_file}")

    return logger
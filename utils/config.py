import yaml
import os
import argparse


class Config:
    """
    配置管理类
    支持从 YAML 加载配置，并允许以属性方式访问 (config.batch_size)
    """

    def __init__(self, config_path=None, args=None):
        # 1. 加载默认配置字典
        self._config_dict = {}

        # 2. 如果指定了 YAML 文件，从中加载
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_conf = yaml.safe_load(f)
                if yaml_conf:
                    self._config_dict.update(yaml_conf)
            print(f"[Config] Loaded configuration from {config_path}")

        # 3. 如果传入了 argparse 的参数，覆盖 YAML 中的同名参数
        # (优先级: 命令行 args > YAML 文件 > 默认值)
        if args:
            args_dict = vars(args)
            for key, value in args_dict.items():
                if value is not None:  # 只有非空参数才覆盖
                    self._config_dict[key] = value

    def __getattr__(self, name):
        """允许使用 config.key 的方式访问字典"""
        return self._config_dict.get(name, None)

    def __repr__(self):
        return str(self._config_dict)

    def get(self, key, default=None):
        return self._config_dict.get(key, default)


def get_config(args=None):
    """
    辅助函数: 获取最终配置对象
    如果 args 中包含 config_file 路径，优先使用
    """
    config_path = None
    if args and hasattr(args, 'config') and args.config:
        config_path = args.config

    return Config(config_path, args)


# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/config_ma.yaml')
    parser.add_argument('--batch_size', type=int, default=None)  # 如果这里传了值，会覆盖 yaml
    args = parser.parse_args()

    cfg = get_config(args)
    print("Batch Size:", cfg.batch_size)
    print("Lesion Type:", cfg.lesion_type)
import yaml

# 默认配置
default_config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 5
}

# 从 YAML 文件读取配置
with open("../config.yaml", "r") as f:
    file_config = yaml.safe_load(f)

# 合并配置（文件配置优先）
config = {**default_config, **file_config}
print(config) 
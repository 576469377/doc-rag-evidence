#!/usr/bin/env python3
"""
从 app.yaml 读取配置的工具脚本
用于shell脚本动态获取配置参数
"""
import sys
import yaml
from pathlib import Path

def load_config():
    """加载 app.yaml 配置"""
    config_path = Path(__file__).parent.parent / "configs" / "app.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_value(config, key_path):
    """通过点号路径获取配置值，例如 'ocr.model_path'"""
    keys = key_path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value

def main():
    if len(sys.argv) < 2:
        print("Usage: config_loader.py <key_path>", file=sys.stderr)
        print("Example: config_loader.py ocr.model_path", file=sys.stderr)
        sys.exit(1)
    
    key_path = sys.argv[1]
    config = load_config()
    value = get_value(config, key_path)
    
    if value is None:
        print(f"Error: Key not found: {key_path}", file=sys.stderr)
        sys.exit(1)
    
    # 输出值（shell脚本可以通过命令替换捕获）
    print(value)

if __name__ == "__main__":
    main()

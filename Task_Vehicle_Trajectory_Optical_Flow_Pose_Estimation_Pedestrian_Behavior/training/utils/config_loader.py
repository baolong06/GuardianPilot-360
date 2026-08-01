import os
import yaml

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../configs/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_config_for_task(task_name):
    config = load_config()
    return config.get(task_name, {})

def get_paths():
    config = load_config()
    return config.get('paths', {})
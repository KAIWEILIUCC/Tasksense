import json
import os

import yaml

from src.execution.dynamic_plan_adaptation.metrics import \
    map_from_function_name_to_function


def get_alternative_pools():
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    alternative_pools_path = config["execution_settings"]["dynamic_plan_adaptation_settings"][dataset_name]["alternative_pools_path"]
    
    alternative_pools = json.load(open(alternative_pools_path, "r"))
    
    return alternative_pools


def get_map_from_alternative_path_to_metrics():
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    get_map_from_alternative_path_to_metrics_path = config["execution_settings"]["dynamic_plan_adaptation_settings"][dataset_name]["map_from_alternative_path_to_metrics_path"]
    
    map_from_alternative_path_to_metrics = json.load(open(get_map_from_alternative_path_to_metrics_path, "r"))
    
    for alternative_path in map_from_alternative_path_to_metrics:
        map_from_alternative_path_to_metrics[alternative_path] = map_from_function_name_to_function[map_from_alternative_path_to_metrics[alternative_path]]
        
    return map_from_alternative_path_to_metrics
    
        
    
def get_map_from_alternative_path_to_threshold():
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    get_map_from_alternative_path_to_threshold_path = config["execution_settings"]["dynamic_plan_adaptation_settings"][dataset_name]["map_from_alternative_path_to_threshold_path"]
    
    map_from_alternative_path_to_threshold = json.load(open(get_map_from_alternative_path_to_threshold_path, "r"))
    
    for alternative_path in map_from_alternative_path_to_threshold:
        map_from_alternative_path_to_threshold[alternative_path] = eval(map_from_alternative_path_to_threshold[alternative_path])
    
    return map_from_alternative_path_to_threshold







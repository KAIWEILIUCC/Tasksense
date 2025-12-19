import json
import os

import yaml

from src.api.llm_apis import send_request
from TaskSense.src.responding.formatter import format_results_to_dict
from src.utils import replace_slot


def generate_response(original_execution_results, user_input, disabled_time_range):
    # read yaml file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    use_dynamic_plan_adaptation = config["execution_settings"]["use_dynamic_plan_adaptation"]
    
    # load the system prompt
    system_prompt = config["responding_settings"]["system_prompt"]
    # load the user prompt
    user_prompt = config["responding_settings"]["user_prompt"]
    # load the example prompt
    example_prompt = open(config["responding_settings"]["example_prompt"], "r").read()
    
    # format the results
    if not use_dynamic_plan_adaptation:
        formatted_results = format_results_to_dict(original_execution_results)
    else:
        formatted_results = original_execution_results
    
    example_prompt = replace_slot(example_prompt, {
        "input": user_input,
        "results": formatted_results,
        "disabled_time_range": disabled_time_range
    })
    
    messages = json.loads(example_prompt)
    messages.insert(0, {
        "role": "system",
        "content": system_prompt
    })
    messages.append({
        "role": "user",
        "content": user_prompt
    })
    
    return send_request(messages)


import asyncio
import json
import os

import boto3
import fastapi_poe as fp
import yaml
from openai import AzureOpenAI
from volcenginesdkarkruntime import Ark

# Volcengine
def send_request_volcengine_api(
    messages,
    llm_name,
    tmp,
):
    # load the config
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    temperature = config["api_settings"]["volcengine"]["temperature"]
    api_key = config["api_settings"]["volcengine"]["api_key"]
    model = config["api_settings"]["volcengine"]["model"]
    
    client = Ark(api_key = api_key)
    
    completion = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature=temperature
    )
    
    return completion.choices[0].message.content

# Azure
def send_request_azure_api(
    messages,
    llm_name,
    temperature=None,
):
    '''
    This function will send a request to the Azure API

    Args:
        messages (list): list of messages
        llm_name (str): the name of the LLM to send the request to
        temperature (int, optional): temperature for the LLM API. The higher the temperature, the more creative the response. Defaults to None.

    Returns:
        str: the response from the LLM API
    '''
    
    # Read the config file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    
    # Load the Azure API settings
    api_key = config["api_settings"]["azure"]["api_key"]
    api_version = config["api_settings"]["azure"]["api_version"]
    azure_endpoint = config["api_settings"]["azure"]["azure_endpoint"]
    temperature = config["api_settings"]["temperature"]
    
    # Create the AzureOpenAI object
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint
    )
    
    # Send the request to the Azure API
    response = client.chat.completions.create(
        model=llm_name,
        messages=messages,
        temperature=temperature
    )
    
    return response.choices[0].message.content

# Bedrock
def prompt_covert_llama3(message):
    prompt = """<|begin_of_text|>"""
    for item in message:
        role = item["role"]
        content = item["content"]
        prompt += f"""<|start_header_id|>{role}<|end_header_id|>
                    {content}
                    <|eot_id|>"""
    prompt += """<|start_header_id|>assistant<|end_header_id|>"""
    return prompt

def send_request_bedrock_api(
    messages,
    llm_name,
    temperature=None,
):
    '''
    This function will send a request to the Bedrock API

    Args:
        messages (list): list of messages
        llm_name (str): the name of the LLM to send the request to
        temperature (int, optional): temperature for the LLM API. The higher the temperature, the more creative the response. Defaults to None.

    Returns:
        str: the response from the LLM API
    '''
    
    # Read the config file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    
    # Load the Bedrock API settings
    api_key = config["api_settings"]["bedrock"]["api_key"]
    region_name = config["api_settings"]["bedrock"]["region_name"]
    
    # Create a Bedrock Runtime client in the specified AWS Region.
    client = boto3.client("bedrock-runtime", region_name=region_name)
    
    if llm_name == "meta.llama3-70b-instruct-v1:0":
        max_gen_len=2048
        top_p=0.9
        
        # convert the message to the Bedrock Llama3 format
        prompt = prompt_covert_llama3(messages)
        
        # construct the request
        request = {
            "prompt": prompt,
            "max_gen_len": max_gen_len,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        # get the response from the Bedrock API
        model_response = client.invoke_model(
            body=json.dumps(request),
            modelId=llm_name
        )
        
        response = model_response["generation"].strip()
        
    return response

# POE
def prompt_covert_poe(message):
    '''
    This function will convert the message to the POE format

    Args:
        message (list): list of messages

    Returns:
        list: the message in the POE format
    '''
    prompt = []
    for item in message:
        role = item["role"]
        if role == "assistant":
            role="bot"
            
        content = item["content"]
        prompt.append(fp.ProtocolMessage(role=role, content=content))
    return prompt

async def get_responses(
    messages,
    llm_name,
    temperature
):
    '''
    Create an asynchronous function to encapsulate the async for loop

    Args:
        api_key (str): the API key for the POE API
        messages (list): list of messages
        model_id (str): the model ID for the POE API
        temperature (int): temperature for the POE API, which controls the creativity of the response
    '''
    
    # Read the config file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    
    # Load the POE API settings
    api_key = config["api_settings"]["poe"]["api_key"]
    
    # define the response variable
    response=""""""
    # for each partial response from the POE API
    async for partial in fp.get_bot_response(messages=messages, bot_name=llm_name, api_key=api_key,temperature=temperature):
        # concatenate the partial response to the response variable
        response=response+partial.text
        
    return response

def send_request_poe_api(
    messages,
    llm_name,
    temperature=None,
):
    '''
    This function will send a request to the POE API

    Args:
        messages (list): list of messages
        llm_name (str): the name of the LLM to send the request to
        temperature (int, optional): temperature for the LLM API. The higher the temperature, the more creative the response. Defaults to None.

    Returns:
        str: the response from the LLM API
    '''
    
    # convert the message to the POE format
    converted_messages = prompt_covert_poe(messages)
    
    # get the API key from the config file
    response=asyncio.run(get_responses(
        converted_messages,
        llm_name,
        temperature
    ))
    
    return response

# GATE
def send_request(messages):
    '''
    This function will send a request to the LLM API, playing the role of a gate to the LLM APIs

    Args:
        messages (list): list of messages
        llm_name (str): the name of the LLM API
        temperature (int, optional): temperature for the LLM API. The higher the temperature, the more creative the response. Defaults to None.
    '''
    
    # Read the config file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    
    # Load the LLM settings
    llm_name = config["api_settings"]["llm_name"]
    temperature = config["api_settings"]["temperature"]
    
    # Map the LLM name to the LLM API
    llm_name_to_llm_api = {
        "gpt_4_32k": send_request_azure_api,
        "gpt-4o": send_request_azure_api,
        "Claude-3-Opus": send_request_poe_api,
        "Claude-3.5-Sonnet": send_request_poe_api,
        "meta.llama3-70b-instruct-v1:0": send_request_bedrock_api,
        "deepseek-v3": send_request_volcengine_api
    }
    
    # send the request to the LLM API
    return llm_name_to_llm_api[llm_name](
        messages,
        llm_name,
        temperature,
    )
    
    
import json
import os

import cohere
import numpy as np
import yaml
from annoy import AnnoyIndex

from src.utils import load_cache_with_lock, save_embedding_cache_with_lock


def get_mostly_related_examples(
    user_input,
    examples,
    number_of_selected_seed_examples
):
    '''
    This function will select the most related examples from the example library

    Args:
        user_input (str): the message string from the user
        examples (str): the examples from the example library
        number_of_selected_seed_examples (int): the number of seed examples to select from the example library

    Returns:
        list: the selected examples from the example library. A list of json objects
    '''
    
    # read config file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    
    # load the POE API settings
    api_key = config["api_settings"]["cohere"]["api_key"]
    
    # load the embedding cache
    embedding_cache_path = config["execution_settings"]["embedding_cache_path"]
    embedding_cache = load_cache_with_lock(embedding_cache_path)
    
    # set up the cohere client
    client = cohere.Client(api_key)
    
    # put the user input in each example into a list
    user_input_list = []
    for example in examples:
        if example["role"] == "user":
            user_input_list.append(example["content"])
    
    # does cache contains all the user inputs?
    cache_contains_all_user_inputs = True
    for example_user_input in user_input_list:
        if example_user_input not in embedding_cache.keys():
            cache_contains_all_user_inputs = False
            break
        
    # get the embeddings of the examples
    if not cache_contains_all_user_inputs:
        examples_embedding = client.embed(
            texts=user_input_list,
            model="embed-english-v3.0",
            input_type="search_document"
        ).embeddings
        
        for i in range(len(examples_embedding)):
            embedding_cache[user_input_list[i]] = examples_embedding[i]
            save_embedding_cache_with_lock(
                embedding_cache_path,
                embedding_cache,
                user_input_list[i]
            )
    else:
        examples_embedding = [
            embedding_cache[example_user_input]
            for example_user_input in user_input_list
        ]
            
    # get the embeddings of the user input
    if user_input not in embedding_cache.keys():
        user_input_embedding = client.embed(
            texts=[user_input],
            model="embed-english-v3.0",
            input_type="search_document"
        ).embeddings[0]
        
        embedding_cache[user_input] = user_input_embedding
        
        save_embedding_cache_with_lock(
            embedding_cache_path,
            embedding_cache,
            user_input
        )
    else:
        user_input_embedding = embedding_cache[user_input]
    
    # build search index
    search_index = AnnoyIndex(np.array(examples_embedding).shape[1], 'angular')
    for i in range(len(examples_embedding)):
        search_index.add_item(i, examples_embedding[i])
    search_index.build(10)
    search_index.save('search_index.ann')
    
    # search the most related examples
    most_related_example_idxes = search_index.get_nns_by_vector(
        user_input_embedding,
        number_of_selected_seed_examples,
        include_distances=True
    )
    
    selected_examples = []
    # add the most realted examples to the selected example list
    for idx in most_related_example_idxes[0][::-1]:
        selected_examples.append(examples[idx * 2])
        selected_examples.append(examples[idx * 2 + 1])
    
    return selected_examples

def get_mostly_related_examples_for_each_category(
    user_input,
    seed_examples,
    augmented_examples,
    number_of_selected_examples
):
    '''
    This function wiil selected the most related seed examples and corresponding augmented examples for a specific example category

    Args:
        user_input (str): the message string from the user
        seed_examples (list): the examples from the seed example library
        augmented_examples (list): the examples from the an augmented example library
        number_of_selected_examples (int): the number of seed examples to select from the example library

    Returns:
        list: the selected examples from the example library. A list of json objects. It includes the seed examples and augmented examples
    '''
    
    # read the config file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    
    # load the POE API settings
    api_key = config["api_settings"]["cohere"]["api_key"]
    
    # load the embedding cache
    embedding_cache_path = config["execution_settings"]["embedding_cache_path"]
    embedding_cache = load_cache_with_lock(embedding_cache_path)
    
    # set up the cohere client
    client = cohere.Client(api_key)
    
    # put the user input in the each seed example into a list
    seed_example_user_input_list = []
    for seed_example in seed_examples:
        if seed_example["role"] == "user":
            seed_example_user_input_list.append(seed_example["content"])
    
    # put the user input in the each augmented example into a list
    augmented_example_user_input_list = []
    for augmented_example in augmented_examples:
        if augmented_example["role"] == "user":
            augmented_example_user_input_list.append(augmented_example["content"])
    
    # does cache contains all the user inputs?
    cache_contains_all_user_inputs_in_seed_examples = True
    for seed_example_user_input in seed_example_user_input_list:
        if seed_example_user_input not in embedding_cache.keys():
            cache_contains_all_user_inputs_in_seed_examples = False
            break
        
    cache_contains_all_user_inputs_in_augmented_examples = True
    for augmented_example_user_input in augmented_example_user_input_list:
        if augmented_example_user_input not in embedding_cache.keys():
            cache_contains_all_user_inputs_in_augmented_examples = False
            break
    
    
    # get the embeddings of the user input
    if user_input not in embedding_cache.keys():
        user_input_embedding = client.embed(
            texts=[user_input],
            model="embed-english-v3.0",
            input_type="search_document"
        ).embeddings[0]
        embedding_cache[user_input] = user_input_embedding
        save_embedding_cache_with_lock(
            embedding_cache_path,
            embedding_cache,
            user_input
        )
    else:
        user_input_embedding = embedding_cache[user_input]
    
    # get the embeddings of the seed examples
    if not cache_contains_all_user_inputs_in_seed_examples:
        seed_examples_embedding = client.embed(
            texts=seed_example_user_input_list,
            model="embed-english-v3.0",
            input_type="search_document"
        ).embeddings
        
        for i in range(len(seed_examples_embedding)):
            embedding_cache[seed_example_user_input_list[i]] = seed_examples_embedding[i]
            save_embedding_cache_with_lock(
                embedding_cache_path,
                embedding_cache,
                seed_example_user_input_list[i]
            )
    else:
        seed_examples_embedding = [
            embedding_cache[seed_example_user_input]
            for seed_example_user_input in seed_example_user_input_list
        ]
    
    # get the embeddings of the augmented examples
    if not cache_contains_all_user_inputs_in_augmented_examples:
        augmented_examples_embedding = client.embed(
            texts=augmented_example_user_input_list,
            model="embed-english-v3.0",
            input_type="search_document"
        ).embeddings
        
        for i in range(len(augmented_examples_embedding)):
            embedding_cache[augmented_example_user_input_list[i]] = augmented_examples_embedding[i]
            save_embedding_cache_with_lock(
                embedding_cache_path,
                embedding_cache,
                augmented_example_user_input_list[i]
            )
    else:
        augmented_examples_embedding = [
            embedding_cache[augmented_example_user_input]
            for augmented_example_user_input in augmented_example_user_input_list
        ]
    
    # build search index for seed examples
    seed_search_index = AnnoyIndex(np.array(seed_examples_embedding).shape[1], 'angular')
    for i in range(len(seed_examples_embedding)):
        seed_search_index.add_item(i, seed_examples_embedding[i])
    seed_search_index.build(10)
    seed_search_index.save('seed_search_index.ann')
    
    # build search index for augmented examples
    augmented_search_index = AnnoyIndex(np.array(augmented_examples_embedding).shape[1], 'angular')
    for i in range(len(augmented_examples_embedding)):
        augmented_search_index.add_item(i, augmented_examples_embedding[i])
    augmented_search_index.build(10)
    augmented_search_index.save('augmented_search_index.ann')
    
    # search the most related seed examples
    most_related_seed_example_idxes = seed_search_index.get_nns_by_vector(
        user_input_embedding,
        number_of_selected_examples,
        include_distances=True
    )
    
    # search the most related augmented examples
    most_related_augmented_example_idxes = augmented_search_index.get_nns_by_vector(
        user_input_embedding,
        number_of_selected_examples,
        include_distances=True
    )
    
    # add the most related seed examples to the selected example list
    selected_seed_examples = []
    
    for idx in most_related_seed_example_idxes[0][::-1]:
        selected_seed_examples.append(seed_examples[idx * 2])
        selected_seed_examples.append(seed_examples[idx * 2 + 1])
    
    # add the most related augmented examples to the selected example list
    selected_augmented_examples = []
    for idx in most_related_augmented_example_idxes[0][::-1]:
        selected_augmented_examples.append(augmented_examples[idx * 2])
        selected_augmented_examples.append(augmented_examples[idx * 2 + 1])
        
    return selected_augmented_examples + selected_seed_examples

def get_examples(
    user_input,
    number_of_selected_seed_examples,
    number_of_selected_seed_examples_per_category
):
    '''
    This function will get the examples from the example library

    Args:
        user_input (str): the message string from the user
        number_of_selected_seed_examples (int): the number of seed examples to select from the example library
        number_of_example_categories (int): the number of categories of examples stored in the example library
        number_of_selected_seed_examples_per_category (int): the number of examples to select from each category, which is calculated based on the number of selected seed examples and the number of example categories
                                                             if its value is 0, indicating that retrival 0 exampel for planning 
    
    Returns:
        list: the selected examples from the example library. A list of json objects
    '''
    
    # read yaml file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    
    dataset_name = config["dataset_name"]
    scale_of_seed_examples = config["planning_settings"]["examples"]["scale_of_seed_examples"]
    
    # paths of the example library
    seed_examples_cls1_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "seed_examples_cls1.json")
    seed_examples_cls2_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "seed_examples_cls2.json")
    seed_examples_cls3_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "seed_examples_cls3.json")
    seed_examples_cls4_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "seed_examples_cls4.json")
    
    derived_examples_cls1_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "derived_examples_cls1.json")
    derived_examples_cls2_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "derived_examples_cls2.json")
    derived_examples_cls3_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "derived_examples_cls3.json")
    derived_examples_cls4_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "derived_examples_cls4.json")
    
    # read examples from the example library
    seed_examples_cls1 = json.load(open(seed_examples_cls1_path, "r"))
    seed_examples_cls2 = json.load(open(seed_examples_cls2_path, "r"))
    seed_examples_cls3 = json.load(open(seed_examples_cls3_path, "r"))
    seed_examples_cls4 = json.load(open(seed_examples_cls4_path, "r"))
    
    derived_examples_cls1 = json.load(open(derived_examples_cls1_path, "r"))
    derived_examples_cls2 = json.load(open(derived_examples_cls2_path, "r"))
    derived_examples_cls3 = json.load(open(derived_examples_cls3_path, "r"))
    derived_examples_cls4 = json.load(open(derived_examples_cls4_path, "r"))
    if number_of_selected_seed_examples_per_category !=0: # [MARKER]: add by xll, for the expriement of impact_of_retrival_seed_size
        # select the most realted examples for each category seperately
        selected_examples_cls1 = get_mostly_related_examples_for_each_category(user_input, seed_examples_cls1, derived_examples_cls1, number_of_selected_seed_examples_per_category)
        selected_examples_cls2 = get_mostly_related_examples_for_each_category(user_input, seed_examples_cls2, derived_examples_cls2, number_of_selected_seed_examples_per_category)
        selected_examples_cls3 = get_mostly_related_examples_for_each_category(user_input, seed_examples_cls3, derived_examples_cls3, number_of_selected_seed_examples_per_category)
        selected_examples_cls4 = get_mostly_related_examples_for_each_category(user_input, seed_examples_cls4, derived_examples_cls4, number_of_selected_seed_examples_per_category)
        
        # select the most related examples from all categories
        selected_examples = get_mostly_related_examples(user_input, selected_examples_cls1+selected_examples_cls2+selected_examples_cls3+selected_examples_cls4, number_of_selected_seed_examples*2)
    
    else:
        selected_examples=[]
    
    return selected_examples

def get_examples_wo_category(
    user_input,
    number_of_selected_seed_examples,
):
    '''
    This function will get the examples from the example library without considering the example categories
    Args:
        user_input (str): the message string from the user
        number_of_selected_seed_examples (int): the number of seed examples to select from the example library
    Returns:
        list: the selected examples from the example library. A list of json objects
    '''
    
    # read yaml file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    scale_of_seed_examples = config["planning_settings"]["examples"]["scale_of_seed_examples"]
    
    # paths of the example library
    seed_examples_cls1_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "seed_examples_cls1.json")
    seed_examples_cls2_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "seed_examples_cls2.json")
    seed_examples_cls3_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "seed_examples_cls3.json")
    seed_examples_cls4_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "seed_examples_cls4.json")
    
    derived_examples_cls1_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "derived_examples_cls1.json")
    derived_examples_cls2_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "derived_examples_cls2.json")
    derived_examples_cls3_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "derived_examples_cls3.json")
    derived_examples_cls4_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "derived_examples_cls4.json")
    
    # read examples from the example library
    seed_examples_cls1 = json.load(open(seed_examples_cls1_path, "r"))
    seed_examples_cls2 = json.load(open(seed_examples_cls2_path, "r"))
    seed_examples_cls3 = json.load(open(seed_examples_cls3_path, "r"))
    seed_examples_cls4 = json.load(open(seed_examples_cls4_path, "r"))
    
    derived_examples_cls1 = json.load(open(derived_examples_cls1_path, "r"))
    derived_examples_cls2 = json.load(open(derived_examples_cls2_path, "r"))
    derived_examples_cls3 = json.load(open(derived_examples_cls3_path, "r"))
    derived_examples_cls4 = json.load(open(derived_examples_cls4_path, "r"))
    
    # merge the seed examples and derived examples
    seed_examples = seed_examples_cls1 + seed_examples_cls2 + seed_examples_cls3 + seed_examples_cls4
    augmented_examples = derived_examples_cls1 + derived_examples_cls2 + derived_examples_cls3 + derived_examples_cls4
    
    selected_examples = get_mostly_related_examples_for_each_category(
        user_input,
        seed_examples,
        augmented_examples,
        number_of_selected_seed_examples
    )
    
    return selected_examples
    
def get_mostly_related_examples_by_rag(
    user_input,
    examples,
    number_of_selected_examples
):
    '''
    This function will select the most related examples from the example library using RAG model

    Args:
        user_input (str): the message string from the user
        examples (str): the examples from the example library
        number_of_selected_examples (int): the number of seed examples to select from the example library

    Returns:
        list: the selected examples from the example library. A list of json objects. It includes the seed examples and augmented examples
    '''
    
    # read the config file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    
    # load the RAG API settings
    api_key = config["api_settings"]["cohere"]["api_key"]
    
    # set up the cohere client
    client = cohere.Client(api_key)
    
    # put the user input in the each example into a list
    example_user_input_list = []
    for example in examples:
        if example["role"] == "user":
            example_user_input_list.append(example["content"])
            
    # get the embeddings of the user input
    user_input_embedding = client.embed(
        texts=[user_input],
        model="embed-english-v3.0",
        input_type="search_document"
    ).embeddings[0]
    
    # get the embeddings of the examples
    examples_embedding = client.embed(
        texts=example_user_input_list,
        model="embed-english-v3.0",
        input_type="search_document"
    ).embeddings
    
    # build search index
    search_index = AnnoyIndex(np.array(examples_embedding).shape[1], 'angular')
    for i in range(len(examples_embedding)):
        search_index.add_item(i, examples_embedding[i])
    search_index.build(10)
    search_index.save('merged_search_index.ann')
    
    # search the most related examples
    most_related_example_idxes = search_index.get_nns_by_vector(
        user_input_embedding,
        number_of_selected_examples,
        include_distances=True
    )
    
    selected_examples = []
    # add the most realted examples to the selected example list
    for idx in most_related_example_idxes[0][::-1]:
        selected_examples.append(examples[idx * 2])
        selected_examples.append(examples[idx * 2 + 1])
        
    return selected_examples

def get_examples_by_rag(
    user_input,
    number_of_selected_examples,
):
    '''
    This function will get the examples from the example library using RAG model

    Args:
        user_input (str): the message string from the user
        number_of_selected_examples (int): the number of seed examples to select from the example library
        
    Returns:
        list: the selected examples from the example library. A list of json objects
    '''
    
    # read yaml file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    
    dataset_name = config["dataset_name"]
    scale_of_seed_examples = 1.0
    
    # paths of the example library
    seed_examples_cls1_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "seed_examples_cls1.json")
    seed_examples_cls2_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "seed_examples_cls2.json")
    seed_examples_cls3_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "seed_examples_cls3.json")
    seed_examples_cls4_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "seed_examples_cls4.json")
    
    derived_examples_cls1_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "derived_examples_cls1.json")
    derived_examples_cls2_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "derived_examples_cls2.json")
    derived_examples_cls3_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "derived_examples_cls3.json")
    derived_examples_cls4_path = os.path.join(config["planning_settings"]["examples"]["example_library_path"], dataset_name, str(scale_of_seed_examples), "derived_examples_cls4.json")
    
    # read examples from the example library
    seed_examples_cls1 = json.load(open(seed_examples_cls1_path, "r"))
    seed_examples_cls2 = json.load(open(seed_examples_cls2_path, "r"))
    seed_examples_cls3 = json.load(open(seed_examples_cls3_path, "r"))
    seed_examples_cls4 = json.load(open(seed_examples_cls4_path, "r"))
    
    derived_examples_cls1 = json.load(open(derived_examples_cls1_path, "r"))
    derived_examples_cls2 = json.load(open(derived_examples_cls2_path, "r"))
    derived_examples_cls3 = json.load(open(derived_examples_cls3_path, "r"))
    derived_examples_cls4 = json.load(open(derived_examples_cls4_path, "r"))
    
    # merge the seed examples and derived examples
    seed_examples = seed_examples_cls1 + seed_examples_cls2 + seed_examples_cls3 + seed_examples_cls4
    augmented_examples = derived_examples_cls1 + derived_examples_cls2 + derived_examples_cls3 + derived_examples_cls4
    examples = seed_examples + augmented_examples
    
    # select the most related examples
    selected_examples = get_mostly_related_examples_by_rag(
        user_input, 
        examples, 
        number_of_selected_examples
    )
    
    print("Number of selected examples: ", number_of_selected_examples)
    return selected_examples

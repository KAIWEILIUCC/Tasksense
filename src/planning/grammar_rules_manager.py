import json
import os

import bidict
import networkx as nx
import yaml



def initilize_tool_name_id_mapping():
    '''
    This function will first load the vocabulary set
    Then it will initialize the tool name to integer id mapping
    
    Returns:
        dict: a dictionary where the key is the tool name and the value is the integer id
    
    '''
    
    tool_name_id_mapping = {
        "initial-input (tool calling id < 0)": -1,
    }
    
    # Load the vocabulary set
    config_path = os.environ.get('CONFIG_PATH')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    vocabulary_set_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "vocabulary_set.json")
    vocabulary_set = json.load(open(vocabulary_set_path))
    tool_name_id_mapping_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "tool_name_id_mapping.json")
    
    tool_id = 0
    
    # Initialize the tool name to integer id mapping
    for tool_type in vocabulary_set:
        for tool_info in vocabulary_set[tool_type]:
            tool_name = tool_info["tool_name"]
            tool_name_id_mapping[tool_name] = tool_id
            tool_id += 1
            
    # Save the tool name to integer id mapping into a file
    print(tool_name_id_mapping_path)
    with open(tool_name_id_mapping_path, 'w') as f:
        json.dump(tool_name_id_mapping, f, indent=4)

def initilize_grammar_rules():
    '''
    This function will initialize the grammar rules
    '''
    
    # The tool name to integer id mapping and the dependencies between tools
    config_path = os.environ.get('CONFIG_PATH')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    tool_name_id_mapping_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "tool_name_id_mapping.json")
    tool_name_id_mapping = json.load(open(tool_name_id_mapping_path))
    tool_dependencies_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "tool_dependencies.json")
    tool_dependencies = json.load(open(tool_dependencies_path))
    print(tool_name_id_mapping)
    
    # add the grammar rules into the graph
    G = nx.DiGraph()
    for start_tool_name, end_tool_name in tool_dependencies:
        G.add_edge(tool_name_id_mapping[start_tool_name], tool_name_id_mapping[end_tool_name])
    
    # Save the graph into a file
    config_path = os.environ.get('CONFIG_PATH')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    grammar_rules_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "grammar_rules.graphml")
    print(grammar_rules_path)
    nx.write_graphml(G, grammar_rules_path)
    
def add_tool(tool_name):
    '''
    This function will add a tool into the grammar rules

    Args:
        tool_name (str): the name of the tool to be added
    '''
    
    # Load the tool name to integer id mapping and the graph
    config_path = os.environ.get('CONFIG_PATH')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    tool_name_id_mapping_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "tool_name_id_mapping.json")
    tool_name_id_mapping = json.load(open(tool_name_id_mapping_path))
    grammar_rules_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "grammar_rules.graphml")
    G = nx.read_graphml(grammar_rules_path)
    
    # Add the tool into the tool name to integer id mapping and the graph
    tool_id = len(tool_name_id_mapping)-1
    tool_name_id_mapping[tool_name] = tool_id
    G.add_node(tool_id)
    
    # Save the tool name to integer id mapping into a file
    with open(tool_name_id_mapping_path, 'w') as f:
        json.dump(tool_name_id_mapping, f, indent=4)
        
    # Save the graph into a file
    nx.write_graphml(G, grammar_rules_path)

def add_dependency(start_tool_name, end_tool_name):
    '''
    This function will add a dependency between two tools (edge) into the grammar rules

    Args:
        start_tool_name (str): the name of the start tool
        end_tool_name (str): the name of the end tool
    '''
    
    # Load the tool name to integer id mapping and the graph
    config_path = os.environ.get('CONFIG_PATH')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    tool_name_id_mapping_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "tool_name_id_mapping.json")
    tool_name_id_mapping = json.load(open(tool_name_id_mapping_path))
    grammar_rules_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "grammar_rules.graphml")
    G = nx.read_graphml(grammar_rules_path)
    
    # check whether the start tool and end tool are in the tool name to integer id mapping
    if start_tool_name not in tool_name_id_mapping:
        add_tool(start_tool_name)
        
    if end_tool_name not in tool_name_id_mapping:
        add_tool(end_tool_name)
        
    # Add the dependency between the two tools into the graph
    G.add_edge(tool_name_id_mapping[start_tool_name], tool_name_id_mapping[end_tool_name])
    
    # Save the graph into a file
    nx.write_graphml(G, grammar_rules_path)
    
def remove_tool(tool_name):
    '''
    This function will remove a tool from the grammar rules

    Args:
        tool_name (str): the name of the tool to be removed
    '''
    
    # Load the tool name to integer id mapping and the graph
    config_path = os.environ.get('CONFIG_PATH')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    tool_name_id_mapping_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "tool_name_id_mapping.json")
    tool_name_id_mapping = json.load(open(tool_name_id_mapping_path))
    grammar_rules_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "grammar_rules.graphml")
    G = nx.read_graphml(grammar_rules_path)
    
    # Remove the tool from the tool name to integer id mapping and the graph
    tool_id = tool_name_id_mapping[tool_name]
    del tool_name_id_mapping[tool_name]
    G.remove_node(tool_id)
    
    # Save the tool name to integer id mapping into a file
    with open(tool_name_id_mapping_path, 'w') as f:
        json.dump(tool_name_id_mapping, f, indent=4)
        
    # Save the graph into a file
    nx.write_graphml(G, grammar_rules_path)
    
def remove_dependency(start_tool_name, end_tool_name):
    '''
    This function will remove a dependency between two tools (edge) from the grammar rules

    Args:
        start_tool_name (str): the name of the start tool
        end_tool_name (str): the name of the end tool
    '''

    # Load the tool name to integer id mapping and the graph
    config_path = os.environ.get('CONFIG_PATH')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    tool_name_id_mapping_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "tool_name_id_mapping.json")
    tool_name_id_mapping = json.load(open(tool_name_id_mapping_path))
    grammar_rules_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "grammar_rules.graphml")
    G = nx.read_graphml(grammar_rules_path)
    
    # Remove the dependency between the two tools from the graph
    G.remove_edge(tool_name_id_mapping[start_tool_name], tool_name_id_mapping[end_tool_name])
    
    # Save the graph into a file
    nx.write_graphml(G, grammar_rules_path)

def get_grammar_rules():
    '''
    This function will return all the grammar rules
    Convert the graph into a list of strings. Each string is a dependency between two tools, like "name of tool1 -> name of tool2"

    Returns:
        list: a list of strings. Each string is a dependency between two tools, like "name of tool1 -> name of tool2"
    '''
    
    # Load the tool name to integer id mapping and the graph
    config_path = os.environ.get('CONFIG_PATH')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    tool_name_id_mapping_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "tool_name_id_mapping.json")
    tool_name_id_mapping = json.load(open(tool_name_id_mapping_path))
    grammar_rules_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "grammar_rules.graphml")
    G = nx.read_graphml(grammar_rules_path)
    
    # convert the tool name to integer id mapping into a bidict
    tool_name_id_mapping = bidict.bidict(tool_name_id_mapping)
    
    # Convert the graph into a list of strings
    grammar_rules = []
    for start_tool_id, end_tool_id in G.edges:
        start_tool_name = tool_name_id_mapping.inverse[int(start_tool_id)]
        end_tool_name = tool_name_id_mapping.inverse[int(end_tool_id)]
        grammar_rules.append(f"{start_tool_name} -> {end_tool_name}")
        
    return grammar_rules

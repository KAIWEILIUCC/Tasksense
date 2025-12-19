import json
import os

import networkx as nx
import yaml

from src.utils import print_tips


# check whether the plan's nodes are correct
def check_node(plan):
    '''
    check whether there is error node in the plan

    Args:
        plan (list): the plan to be checked

    Returns:
        bool: True if there is no error node in the plan, False otherwise
    '''
    
    # read config file
    config_path = os.environ.get('CONFIG_PATH')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    tool_name_id_mapping_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "tool_name_id_mapping.json")
    tool_name_id_mapping = json.load(open(tool_name_id_mapping_path))
    
    # node correct flag
    node_correct = True
    
    # node error collection
    node_error = []
    
    # check whether the tools in the plan are in the tool name to integer id mapping, if not, add them into the node error collection
    for tool in plan:
        if tool["tool_name"] not in tool_name_id_mapping:
            node_correct = False
            node_error.append(tool["tool_name"])
    
    return node_correct, node_error

# check whether the plan's edges are correct
def check_edge(G_rules, G_plan):
    '''
    This function checks whether the inputted plan's edges are correct.

    Args:
        plan_rules (nx.DiGraph): the graph of the grammar rules
        plan (nx.DiGraph): the graph of the plan

    Returns:
        bool: True if the plan's edges are correct, False otherwise
    '''
    
    # read config file
    config_path = os.environ.get('CONFIG_PATH')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    tool_name_id_mapping_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "tool_name_id_mapping.json")
    tool_name_id_mapping = json.load(open(tool_name_id_mapping_path))
    tool_id_name_mapping = dict(zip(tool_name_id_mapping.values(), tool_name_id_mapping.keys()))
    
    # print(tool_id_name_mapping)
    # print(tool_name_id_mapping)
    
    # edge correct flag
    edge_correct = True
    
    # edge error collection
    edge_errors= []
    
    # wrong nodes
    wrong_nodes = []
    
    # check whether the edges in the plan are in the grammar rules, if not, add them into the edge error collection
    
    # print("====================================")
    # print("G_plan.edges: ", G_plan.edges)
    # print("G_rules.edges: ", G_rules.edges)
    # print("====================================")
    
    if not all(edge in G_rules.edges for edge in G_plan.edges):
        edge_correct = False
        
        for edge in G_plan.edges:
            if edge not in G_rules.edges:
                edge_errors.append(tool_id_name_mapping[int(edge[0])]+" -> "+tool_id_name_mapping[int(edge[1])])
                # print(edge)
                # print(G_plan.nodes[edge[1]])
                wrong_nodes.append(G_plan.nodes[edge[1]]["id_in_plan"])
    
    return edge_correct, edge_errors, wrong_nodes

# check whether the plan's grammar is correct
def check_grammar_rules(plan):
    '''
    This function checks whether the inputted plan's grammar is correct.

    Args:
        plan (int): the plan to be checked

    Returns:
        dict: the checking result
    '''
    
    # read config file
    config_path = os.environ.get('CONFIG_PATH')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    
    # read the tool name to integer id mapping
    tool_name_id_mapping_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "tool_name_id_mapping.json")
    tool_name_id_mapping = json.load(open(tool_name_id_mapping_path))
    
    # create an empty directed graph to store the plan
    G_plan = nx.DiGraph()
    
    # read the global grammar rules
    grammar_rules_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "grammar_rules.graphml")
    G_rules = nx.read_graphml(grammar_rules_path)
    
    # initialize the checking result
    checking_result = {
        "pass": None,
        "error": None
    }
    
    # check whether there is error node in the plan
    node_correct, node_error = check_node(plan)
    
    if not node_correct:
        checking_result["pass"] = False
        checking_result["error"] = "The following nodes are not in the grammar rules: "+", ".join(node_error)
        return checking_result
    
    # construct the plan graph
    for tool in plan:
        G_plan.add_node(str(tool_name_id_mapping[tool["tool_name"]]), id_in_plan=tool["id"])
        
        for dep in tool["dep"]:
            if dep >= 0:
                G_plan.add_edge(
                    str(tool_name_id_mapping[plan[dep]["tool_name"]]),
                    str(tool_name_id_mapping[tool["tool_name"]])
                )
            else:
                G_plan.add_edge(
                    str(-1),
                    str(tool_name_id_mapping[tool["tool_name"]])
                )
    
    edge_correct, edge_errors, wrong_nodes = check_edge(G_rules, G_plan)
    
    if edge_correct:
        checking_result["pass"] = True
        # print("[√] The plan's grammar is correct.")
        print_tips("The plan's grammar is correct.", text_color="yellow", emoji="white_check_mark", border=False)
    else:
        checking_result["pass"] = False
        checking_result["error"] = "The following edges are not in the grammar rules: "+str(edge_errors)+". The following nodes are not in the grammar rules: " +str(wrong_nodes)
        
    return checking_result

# check whether the plan's format is correct
def check_format(plan_str):
    '''
    This function checks whether the inputted plan string's format is correct.

    Args:
        plan_str (str): the plan in string form to be checked
        
    Returns:
        dict: the checking result
        plan_in_json: the plan in json form
    '''
    
    # if the plan string contains the word "error"
    if "error" in plan_str:
        return {
            "pass": False,
            "error": "The plan contains the word 'error'."
        }, None
    else:
        print_tips("There is no 'error' word in the response.", text_color="yellow", emoji="white_check_mark", border=False)
    
    # Remove leading and trailing ' ' and '\n' characters
    plan_str = plan_str.strip()
    
    # try to load the plan string into a json object
    try:
        plan = json.loads(plan_str)
        print_tips("The plan is in json format.", text_color="yellow", emoji="white_check_mark", border=False)
    except Exception as e:
        return {
            "pass": False,
            "error": "The plan is not in json format."
        }, None
        
    # check whether the plan is empty
    if len(plan) == 0:
        return {
            "pass": True,
            "error": None
        }, []
    else:
        print_tips("The plan is not empty.", text_color="yellow", emoji="white_check_mark", border=False)
    
    # check if the arttributes of each tool are correct
    for tool in plan:
        if "args" not in tool.keys():
            return {
                "pass": False,
                "error": "The tool does not have the 'args' attribute."
            }, None
        elif "dep" not in tool.keys():
            return {
                "pass": False,
                "error": "The tool does not have the 'dep' attribute."
            }, None
        elif "id" not in tool.keys():
            return {
                "pass": False,
                "error": "The tool does not have the 'id' attribute."
            }, None
        elif "tool_name" not in tool.keys():
            return {
                "pass": False,
                "error": "The tool does not have the 'tool_name' attribute."
            }, None
        else:
            print_tips(f"The tool {tool['tool_name']} has all the necessary attributes.", text_color="yellow", emoji="white_check_mark", border=False)
    
    return {
        "pass": True,
        "error": None
    }, plan
    
# check whether the plan is correct
def check_plan(plan_str):
    '''
    This function checks whether the inputted plan is correct.

    Args:
        plan_str (str): the plan to be checked

    Returns:
        dict: the checking result
    '''
    
    # check whether the plan's format is correct
    format_check_result, plan = check_format(plan_str)
    
    if plan == []:
        return {
            "pass": True,
            "error": None
        }
    
    if not format_check_result["pass"]:
        return format_check_result
    
    # check whether the plan's grammar is correct
    grammar_check_result = check_grammar_rules(plan)
    
    if not grammar_check_result["pass"]:
        return grammar_check_result
    
    return {
        "pass": True,
        "error": None
    }


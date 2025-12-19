import copy
import random
from collections import Counter

from src.execution.dynamic_plan_adaptation.alternative_path_manager import (
    get_alternative_pools, get_map_from_alternative_path_to_metrics,
    get_map_from_alternative_path_to_threshold)
from src.execution.toolbox import tool_name_tool_implementation_map
from src.utils import print_tips


def extract_dependencies(plan):
    '''
    This function is to extract the dependencies from the plan

    Args:
        plan (list): the plan
        
    Returns:
        list: extracted dependencies
    '''
    
    dependencies = []
    
    for tool_calling in plan:
        for dep_id in tool_calling["dep"]:
            if dep_id >= 0:
                dependencies.append(plan[dep_id]["tool_name"] + " -> " + tool_calling["tool_name"])
            else:
                dependencies.append("initial input -> " + tool_calling["tool_name"])
                
    return dependencies

def match(alternative_path, orriginal_plan):
    '''
    This dunction is to match the alternative path to the original plan

    Args:
        alternative_path (list): the alternative path
        orriginal_plan (list): the original plan
    '''         
    # judge the relationship between the alternative path and the original plan
    all_tool_calling_ids_of_original_plan = []
    all_tool_calling_ids_of_alternative_path = []
    
    non_ending_tool_calling_ids_of_original_plan = []
    non_ending_tool_calling_ids_of_alternative_path = []
    
    ending_tool_calling_ids_of_original_plan = []
    ending_tool_calling_ids_of_alternative_path = []
    ending_tool_names_of_original_plan = []
    ending_tool_names_of_alternative_path = []
    
    for tool_calling in orriginal_plan:
        all_tool_calling_ids_of_original_plan.append(tool_calling["id"])
        for dep_id in tool_calling["dep"]:
            if dep_id >= 0:
                non_ending_tool_calling_ids_of_original_plan.append(dep_id)
                
    for tool_calling in alternative_path:
        all_tool_calling_ids_of_alternative_path.append(tool_calling["id"])
        for dep_id in tool_calling["dep"]:
            if dep_id >= 0:
                non_ending_tool_calling_ids_of_alternative_path.append(dep_id)
                
    for tool_calling in orriginal_plan:
        if tool_calling["id"] not in non_ending_tool_calling_ids_of_original_plan:
            ending_tool_calling_ids_of_original_plan.append(tool_calling["id"])
            ending_tool_names_of_original_plan.append(tool_calling["tool_name"])
            
    for tool_calling in alternative_path:
        if tool_calling["id"] not in non_ending_tool_calling_ids_of_alternative_path:
            ending_tool_calling_ids_of_alternative_path.append(tool_calling["id"])
            ending_tool_names_of_alternative_path.append(tool_calling["tool_name"])
    
    dependencies_of_original_plan = extract_dependencies(orriginal_plan)
    dependencies_of_alternative_path = extract_dependencies(alternative_path)
    relationship = None
    matching_degree = None
    
    # print("\n\n\n")
    # print(Counter(dependencies_of_original_plan))
    # print(Counter(dependencies_of_alternative_path))
    
    # 1. 'equal' relationship. Condition: the dependencies within the original plan and the alternative path are the same
    if Counter(dependencies_of_original_plan) == Counter(dependencies_of_alternative_path) and ending_tool_names_of_alternative_path == ending_tool_names_of_original_plan:
        relationship = "equal"
        matching_degree = 1.0
    # 2. 'subplan' relationship. Condition: all the dependencies within the alternative path are included in the original plan
    elif set(dependencies_of_alternative_path).issubset(set(dependencies_of_original_plan)) and set(ending_tool_names_of_alternative_path).issubset(set(ending_tool_names_of_original_plan)):
        relationship = "subplan"
        matching_degree = len(dependencies_of_alternative_path) / len(dependencies_of_original_plan)
    # 3. 'invalid' relationship. Condition: the dependencies within the alternative path are not included in the original plan
    else:
        relationship = "invalid"
        matching_degree = 0.0
        
    assert relationship is not None, "The relationship between the alternative path and the original plan is not determined."
    assert matching_degree is not None, "The matching degree between the alternative path and the original plan is not determined."
    
    return relationship, matching_degree

def get_tool_calling_by_id(plan, tool_calling_id):
    '''
    This function is to get the tool calling by the id

    Args:
        tool_calling_id (int): the id of the tool calling
        plan (list): the plan
        
    Returns:
        dict: the tool calling with the id
    '''
    
    for tool_calling in plan:
        if tool_calling["id"] == tool_calling_id:
            return tool_calling
        
    return None

def are_equivalent_nodes(tool_calling_id_of_subplan, tool_calling_id_of_original_plan, subplan, original_plan):
    # condition 1: the two nodes have the same tool name
    c1 = None
    
    tool_calling_in_subplan = get_tool_calling_by_id(subplan, tool_calling_id_of_subplan)
    tool_calling_in_original_plan = get_tool_calling_by_id(original_plan, tool_calling_id_of_original_plan)
    
    if tool_calling_in_subplan["tool_name"] == tool_calling_in_original_plan["tool_name"]:
        c1 = True
    else:
        c1 = False
        
    assert c1 is not None, "condition 1 is not determined."
    
    # condition 2: the two nodes have the same input
    # the tool names in the dependency list of the two nodes are the same
    c2 = None
    
    dep_names_of_subplan = [get_tool_calling_by_id(subplan, dep_id)["tool_name"] for dep_id in tool_calling_in_subplan["dep"]]
    dep_names_of_original_plan = [get_tool_calling_by_id(original_plan, dep_id)["tool_name"] for dep_id in tool_calling_in_original_plan["dep"]]
    
    if set(dep_names_of_subplan) == set(dep_names_of_original_plan):
        c2 = True
    else:
        c2 = False
        
    assert c2 is not None, "condition 2 is not determined."
    
    return c1 and c2

def adjust_tool_calling_ids(plan):
    id_map = {}
    new_id = 0
    
    for tool_calling in plan:
        old_id = tool_calling['id']
        tool_calling['id'] = new_id
        id_map[old_id] = new_id
        
        if tool_calling['dep'][0] != -1:
            tool_calling['dep'][0] = id_map[tool_calling['dep'][0]]
        
        for key, value in tool_calling['args'].items():
            if isinstance(value, str) and value.startswith('<GENERATED>-'):
                old_ref_id = int(value.split('-')[1])
                new_ref_id = id_map[old_ref_id]
                tool_calling['args'][key] = f'<GENERATED>-{new_ref_id}'
        
        new_id += 1
    
    return plan

def remove_subplan(subplan, original_plan):
    '''
    This function is to remove the subplan from the original plan

    Args:
        subplan (list): the subplan to be removed from the original plan
        original_plan (dict): the original plan
        
    Returns:
        list: the updated original plan
    '''
    
    identification_tool_names = ["face-recognition-rgb", "identification-depth", "speaker-recognition-audio"]
    
    ending_tool_calling_ids_of_original_plan = []
    ending_tool_calling_ids_of_subplan = []
    ending_tool_names_of_original_plan = []
    ending_tool_names_of_subplan = []

    
    all_tool_calling_ids_of_original_plan = []
    all_tool_calling_ids_of_subplan = []
    
    non_ending_tool_calling_ids_of_original_plan = []
    non_ending_tool_calling_ids_of_subplan = []
    
    # map from tool calling id to out degree of tool calling
    out_degree_of_tool_calling_in_original_plan = {}
    in_degree_of_tool_calling_in_original_plan = {}
    
    # initialize the out degree of tool calling in the original plan
    for tool_calling in original_plan:
        out_degree_of_tool_calling_in_original_plan[tool_calling["id"]] = 0
        in_degree_of_tool_calling_in_original_plan[tool_calling["id"]] = 0
        
    # update the out degree of tool calling in the original plan
    for tool_calling in original_plan:
        for dep_id in tool_calling["dep"]:
            if dep_id >= 0:
                out_degree_of_tool_calling_in_original_plan[dep_id] += 1
                in_degree_of_tool_calling_in_original_plan[tool_calling["id"]] += 1
                if dep_id not in all_tool_calling_ids_of_original_plan:
                    non_ending_tool_calling_ids_of_original_plan.append(dep_id)
    
    for tool_calling in subplan:
        for dep_id in tool_calling["dep"]:
            if dep_id >= 0:
                if dep_id not in all_tool_calling_ids_of_subplan:
                    non_ending_tool_calling_ids_of_subplan.append(dep_id)
                    
    all_tool_calling_ids_of_original_plan = [tool_calling["id"] for tool_calling in original_plan]
    all_tool_calling_ids_of_subplan = [tool_calling["id"] for tool_calling in subplan]
    
    # find the ending tool calling ids of the original plan
    for tool_calling in original_plan:
        if tool_calling["id"] not in non_ending_tool_calling_ids_of_original_plan:
            ending_tool_calling_ids_of_original_plan.append(tool_calling["id"])
            ending_tool_names_of_original_plan.append(tool_calling["tool_name"])
            
    # find the ending tool calling ids of the subplan
    for tool_calling in subplan:
        if tool_calling["id"] not in non_ending_tool_calling_ids_of_subplan:
            ending_tool_calling_ids_of_subplan.append(tool_calling["id"])
            ending_tool_names_of_subplan.append(tool_calling["tool_name"])
    
    assert len(ending_tool_calling_ids_of_subplan) > 0, "There is no ending tool calling id in the subplan."
    assert len(ending_tool_calling_ids_of_original_plan) == len(ending_tool_names_of_original_plan), "The number of ending tool calling ids is not equal to the number of ending tool names in the original plan."
    assert len(ending_tool_calling_ids_of_subplan) == len(ending_tool_names_of_subplan), "The number of ending tool calling ids is not equal to the number of ending tool names in the subplan."
    
    plan_without_subplan = None
    
    # print(len(ending_tool_calling_ids_of_original_plan))
    
    
    # if there is only one ending tool calling id and the tool is identification tool, then the plan without the subplan is an empty list
    if len(ending_tool_calling_ids_of_original_plan) == 1 and ending_tool_names_of_original_plan[0] in identification_tool_names:
        plan_without_subplan = []
        
    # if there are twol tool calling id and one of them is identification tool and the other is not, then the plan without the subplan is an empty list
    elif len(ending_tool_calling_ids_of_original_plan) == 2 and\
        ((ending_tool_names_of_original_plan[0] in identification_tool_names and ending_tool_names_of_original_plan[1] not in identification_tool_names) or\
        (ending_tool_names_of_original_plan[0] not in identification_tool_names and ending_tool_names_of_original_plan[1] in identification_tool_names))and\
            (ending_tool_names_of_original_plan[0] != "object-detection-rgb") and (ending_tool_names_of_original_plan[1] != "object-detection-rgb"):
        plan_without_subplan = []
    
    elif len(ending_tool_calling_ids_of_original_plan) == 2 and\
        ((ending_tool_names_of_original_plan[0] in identification_tool_names and ending_tool_names_of_original_plan[1] == "object-detection-rgb") or\
        (ending_tool_names_of_original_plan[0] == "object-detection-rgb" and ending_tool_names_of_original_plan[1] in identification_tool_names)):
        plan_without_subplan = []
        
        assert len(ending_tool_names_of_subplan) == 1, "The number of ending tool names of the subplan is not 1."
        
        # print("ending_tool_names_of_subplan: ", ending_tool_names_of_subplan)
        
        if ending_tool_names_of_subplan[0] not in identification_tool_names:
            
            assert ending_tool_names_of_subplan[0] == "object-detection-rgb", "The ending tool name of the subplan is not object-detection-rgb."
            
            for tool_calling in original_plan:
                if tool_calling["tool_name"] != "object-detection-rgb":
                    plan_without_subplan.append(tool_calling)
        else:
            object_dep_ids = None
            for tool_calling in original_plan:
                if tool_calling["tool_name"] == "object-detection-rgb":
                    object_dep_ids = tool_calling["dep"]
                    break
            
            for tool_calling in original_plan:
                if tool_calling["id"] in object_dep_ids or tool_calling["tool_name"] == "object-detection-rgb":
                    plan_without_subplan.append(tool_calling)
                    
    # other conditions
    else:
        tool_calling_ids_to_be_removed = []
        
        ending_tool_calling_id_to_delete = None
        for ending_tool_calling_id in ending_tool_calling_ids_of_subplan:
            if get_tool_calling_by_id(subplan, ending_tool_calling_id)["tool_name"] not in identification_tool_names:
                for original_ending_tool_calling_id in ending_tool_calling_ids_of_original_plan:
                    if are_equivalent_nodes(ending_tool_calling_id, original_ending_tool_calling_id, subplan, original_plan):
                        ending_tool_calling_id_to_delete = original_ending_tool_calling_id
                        break
                break
        
        found = True
        
        while found:
            found = False
            
            if not get_tool_calling_by_id(original_plan, ending_tool_calling_id_to_delete):
                break
            
            
            
            if out_degree_of_tool_calling_in_original_plan[ending_tool_calling_id_to_delete] == 0 and\
                in_degree_of_tool_calling_in_original_plan[ending_tool_calling_id_to_delete] <= 1:
                    
                tool_calling_ids_to_be_removed.append(ending_tool_calling_id_to_delete)
                ending_tool_calling_id_to_delete = get_tool_calling_by_id(original_plan, ending_tool_calling_id_to_delete)["dep"][0]
                
                if ending_tool_calling_id_to_delete >= 0:
                    out_degree_of_tool_calling_in_original_plan[ending_tool_calling_id_to_delete] -= 1
                
                found = True
        
        plan_without_subplan = [tool_calling for tool_calling in original_plan if tool_calling["id"] not in tool_calling_ids_to_be_removed]
                    
    plan_without_subplan = adjust_tool_calling_ids(plan_without_subplan)
    
    # pdb.set_trace()
    
    return plan_without_subplan
    
def adjust_plan(original_plan):
    '''
    This function is to find the alternative paths of the original plan

    Args:
        original_plan (list): the original plan
        
    Returns:
        dict: map from time interval to feasible alternative list
        list: unexecutable time ranges
    '''
    
    # match the original plan to the alternative pools
    matched_pools = []
    original_plan_copy = copy.deepcopy(original_plan)
    alternative_pools = get_alternative_pools()
    
    # skip the data quality check
    '''
    map_from_alternative_path_to_metrics = get_map_from_alternative_path_to_metrics()
    map_from_alternative_path_to_threshold = get_map_from_alternative_path_to_threshold()
    '''
    
    while len(original_plan_copy) > 0:
        most_mathced_pool = None
        most_mathced_alternative_id = None
        max_matching_degree = 0.0
        
        for alternative_pool in alternative_pools:
            for alternative_id in alternative_pool:
                relationship, matching_degree = match(alternative_pool[alternative_id], original_plan_copy)
                
                # print(alternative_id, matching_degree, alternative_pool[alternative_id])
                
                if relationship in ["equal", "subplan"] and matching_degree > max_matching_degree:
                    most_mathced_pool = alternative_pool
                    most_mathced_alternative_id = alternative_id
                    max_matching_degree = matching_degree
        
        assert most_mathced_pool is not None, "No alternative path is found for the current plan."
        assert most_mathced_alternative_id is not None, "No alternative path is found for the current plan."
        
        matched_pools.append(most_mathced_pool)
        original_plan_copy = remove_subplan(most_mathced_pool[most_mathced_alternative_id], original_plan_copy)
        
    # filter the alternative pool for each time interval
    map_from_time_interval_to_feasible_alternative_list = {}
    map_from_time_interval_to_filtered_alternative_list = {}
    unexecutable_time_ranges = set()
    whole_time_range = original_plan[0]["args"]["text"]
    
    for time_range in whole_time_range:
        date = time_range[0]
        start_hour = time_range[1]
        end_hour = time_range[2]
        
        for time_point in range(start_hour, end_hour+1):
            map_from_time_interval_to_feasible_alternative_list[str(date)+"-"+str(time_point)+'-'+str(time_point)] = []
            
    print_tips("Start filtering alternative paths based on data availability and quality ...", emoji="mag", text_color="yellow", border=False)
    
    # filtering  
    for time_range_str in map_from_time_interval_to_feasible_alternative_list:
        date, start_hour, end_hour = time_range_str.split("-")
        retrival_range = [[int(date), int(start_hour), int(end_hour)]]
        filtered_pools = []
        
        # for each alternative pool
        for alternative_pool in matched_pools:
            filtered_alternative_pool = {}
            
            # for each alternative path in the alternative pool
            for alternative_id in alternative_pool:
                alternative_path = alternative_pool[alternative_id]
                
                # fetch the start tool calling of the alternative path
                start_tool_calling = None
                for tool_calling in alternative_path:
                    if tool_calling["dep"][0] < 0:
                        start_tool_calling = tool_calling
                        break
                    
                assert start_tool_calling is not None, "The start tool calling of the alternative path is not found."
                
                # construct the input to alternative path
                input_for_retrieval = {
                    "tool_name": start_tool_calling["tool_name"],
                    "output": {
                        "pre_sign": [],
                        "post_sign": []
                    },
                    "saving": True,
                    "execute_next": True
                }
                
                input_for_retrieval["output"]["text"] = retrival_range
                for _ in range(len(input_for_retrieval["output"]["text"])):
                    input_for_retrieval["output"]["post_sign"].append(hash(str(random.randint(0, 999999))))
                    
                outputs = tool_name_tool_implementation_map[start_tool_calling["tool_name"]](input_for_retrieval, {})
                
                # check data missing
                if len(outputs["output"]["post_sign"]) == 0:
                    print_tips("No data retrived: {}".format(input_for_retrieval), emoji="x", text_color="red", border=False)
                    continue
                
                # check data quality
                retrieved_data_type = None
                for key in outputs["output"]:
                    if key != "pre_sign" and key != "post_sign":
                        retrieved_data_type = key
                        continue
                
                assert retrieved_data_type is not None, "The retrieved data type is not determined."
                
                map_from_alternative_path_to_metrics = get_map_from_alternative_path_to_metrics()
                map_from_alternative_path_to_threshold = get_map_from_alternative_path_to_threshold()
                data_quality_value = map_from_alternative_path_to_metrics[alternative_id](outputs["output"][retrieved_data_type][0])
                if not map_from_alternative_path_to_threshold[alternative_id](data_quality_value):
                    print_tips("Data quality of {} in {} is not good enough.".format(alternative_id, time_range_str), emoji="x", text_color="red", border=False)
                    continue
                
                filtered_alternative_pool[alternative_id] = copy.deepcopy(alternative_path)
                for i in range(len(filtered_alternative_pool[alternative_id])):
                    if filtered_alternative_pool[alternative_id][i]["dep"][0] < 0:
                        filtered_alternative_pool[alternative_id][i]["args"]["text"] = retrival_range
                        
            filtered_pools.append(filtered_alternative_pool)
            if len(filtered_alternative_pool) == 0:
                unexecutable_time_ranges.add(time_range_str)

        map_from_time_interval_to_filtered_alternative_list[time_range_str] = filtered_pools
    
    keys_to_remove = [key for key in map_from_time_interval_to_filtered_alternative_list if key.endswith('-24')]
    for key in keys_to_remove:
        del map_from_time_interval_to_filtered_alternative_list[key]
    unexecutable_time_ranges = {time_range for time_range in unexecutable_time_ranges if not time_range.endswith('-24')}
        
    print_tips("Finished filtering alternative paths based on data availability and quality.", emoji="mag", text_color="yellow", border=False)
        
    return map_from_time_interval_to_filtered_alternative_list, unexecutable_time_ranges
                
                
                

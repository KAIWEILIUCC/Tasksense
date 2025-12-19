import copy
import json
from collections import Counter

from bidict import bidict

from src.planning.vocabulary_set_manager import (
    get_all_modalities, get_all_vocabulary_set,
    get_all_vocabulary_set_for_tool_type, get_map_from_tool_name_to_modality
)


def adjust_dep_ids(results):
    '''
    This function adjusts the dependent ids of the results

    Args:
        results (dict): The original results from the search query
        
    Returns:
        dict: The results with adjusted dependent ids
    '''
    new_results = {}
    for key in results:
        new_results[key] = results[key]
        if key >= 0 and results[key]["tool_calling_information"]["dep"][0] == -1:
            new_results[key]["tool_calling_information"]["dep"] = [int(-1 * (int(key) + 1))]
    
    return new_results

def remove_result_by_key(results, keys_to_remove):
    if results:
        for key in keys_to_remove:
            if key in results:
                results.pop(key)
            elif str(key) in results:
                results.pop(str(key))

def cut_path(path, tool_index_to_delete):
    path_expressed_by_tool_index = path["path_expressed_by_tool_index"]
    path_expressed_by_sign_of_tool_calling = path["path_expressed_by_sign_of_tool_calling"]
    
    path_expressed_by_tool_index_list = path_expressed_by_tool_index.split("->")
    path_expressed_by_sign_of_tool_calling_list = path_expressed_by_sign_of_tool_calling.split("->")
    
    new_path_expressed_by_tool_index_list = []
    new_path_expressed_by_sign_of_tool_calling_list = []
    
    for i in range(len(path_expressed_by_tool_index_list)):
        if path_expressed_by_tool_index_list[i] not in tool_index_to_delete:
            new_path_expressed_by_tool_index_list.append(path_expressed_by_tool_index_list[i])
            new_path_expressed_by_sign_of_tool_calling_list.append(path_expressed_by_sign_of_tool_calling_list[i])
            
    new_path_expressed_by_tool_index = "->".join(new_path_expressed_by_tool_index_list)
    new_path_expressed_by_sign_of_tool_calling = "->".join(new_path_expressed_by_sign_of_tool_calling_list)
    
    path["path_expressed_by_tool_index"] = new_path_expressed_by_tool_index
    path["path_expressed_by_sign_of_tool_calling"] = new_path_expressed_by_sign_of_tool_calling
    
    return path

def remove_subpath(paths):
    if not paths:
        return []
    
    paths_without_sub_path = []
    
    for i in range(len(paths)):
        path_i = paths[i]["path_expressed_by_sign_of_tool_calling"]
        is_sub_path = False
        
        for j in range(len(paths)):
            if i == j:
                continue
            
            path_j = paths[j]["path_expressed_by_sign_of_tool_calling"]
            
            if path_i != path_j and path_i in path_j:
                is_sub_path = True
                break
        
        if not is_sub_path:
            paths_without_sub_path.append(paths[i])
    
    return paths_without_sub_path


def unify_tool_calling_ids_as_int(original_results):
    '''
    This function unifies the tool_calling ids as integers

    Args:
        original_results (dict): The original results from the search query
        
    Returns:
        dict: The results with unified tool_calling ids
    '''
    new_results = {}
    for key in original_results:
        new_results[int(key)] = original_results[key]
    
    return new_results


def extract_mappings(results):
    '''
    This function extracts the mappings and information from the results

    Args:
        results (dict): The original results

    Returns:
        tuple: (bi_map_from_tool_index_to_tool_name, 
                map_from_tool_index_to_tool_name_inference,
                map_from_tool_calling_id_to_tool_index_inference,
                tool_calling_ids,
                tool_calling_ids_ending,
                tool_calling_ids_non_ending)
    '''
    
    results = unify_tool_calling_ids_as_int(results)
    
    bi_map_from_tool_index_to_tool_name = bidict({"-1": "initial input"})
    whole_vocabulary_set = get_all_vocabulary_set()
    
    tool_index = 0
    for tool_type in whole_vocabulary_set:
        for tool_info in whole_vocabulary_set[tool_type]:
            bi_map_from_tool_index_to_tool_name[str(tool_index)] = tool_info["tool_name"]
            tool_index += 1
            
    map_from_tool_index_to_tool_name_inference = {}
    map_from_tool_calling_id_to_tool_index_inference = {}
    
    tool_calling_ids = list(results.keys())
    tool_calling_ids_non_ending = set()
    
    for tool_calling_id, tool_calling_result in results.items():
        tool_name = tool_calling_result["tool_calling_information"]["tool_name"]
        tool_index = bi_map_from_tool_index_to_tool_name.inv[tool_name]
        
        map_from_tool_index_to_tool_name_inference[tool_index] = tool_name
        map_from_tool_calling_id_to_tool_index_inference[tool_calling_id] = tool_index
        
        dep_list = tool_calling_result["tool_calling_information"]["dep"]
        tool_calling_ids_non_ending.update(dep_list)

    tool_calling_ids_ending = list(set(tool_calling_ids) - tool_calling_ids_non_ending)
    tool_calling_ids_non_ending = list(tool_calling_ids_non_ending)
    
    return (bi_map_from_tool_index_to_tool_name, 
            map_from_tool_index_to_tool_name_inference,
            map_from_tool_calling_id_to_tool_index_inference,
            tool_calling_ids,
            tool_calling_ids_ending,
            tool_calling_ids_non_ending)


def extract_time_stamp_for_path(path):
    start_node = path[-1]
    assert start_node["tool_calling_id"] < 0, "The starting node should be the initial input"
    
    keys_to_delete = ["tool_calling_id", "tool_name", "post_sign", "tool_index"]
    input_list = []
    for key in start_node:
        if key not in keys_to_delete:
            input_list.append(start_node[key])
        
    vocabulary_set_db_tool = get_all_vocabulary_set_for_tool_type("db_tool")
    db_tool_names = [tool["tool_name"] for tool in vocabulary_set_db_tool]
    all_modalities = get_all_modalities()
    
    time_stamp = None
    for node in path:
        if node["tool_name"] in db_tool_names:
            for key in node:
                if key in all_modalities:
                    file_name = node[key]
                    file_name = file_name.split("/")[-1].split(".")[0]
                    date_str = str(input_list[0][0])
                    date_str = date_str[0:4]+"-"+date_str[4:6]+"-"+date_str[6:8]
                    time_stamp = date_str + " " + file_name + ":00" + " -> " + date_str + " " + "{:02d}".format(int(file_name)+1) + ":00"
                    break
                
    return time_stamp
    
def simplify_paths(paths):
    '''
    This function converts the paths into a simpler format

    Args:
        paths (list): the original paths
        
    Returns:
        list: The simplified paths
    '''
    
    simplified_paths = []
    deleted_keys = ["tool_calling_id", "tool_name", "post_sign", "tool_index"]
    
    for path in paths:
        path_expressed_by_tool_index = ""
        path_expressed_by_sign_of_tool_calling = ""
        
        for node_idx in range(len(path)-1, -1, -1):
            if node_idx != 0:
                path_expressed_by_tool_index += str(path[node_idx]["tool_index"]) + "->"
                path_expressed_by_sign_of_tool_calling += str(path[node_idx]["post_sign"]) + "->"
            else:
                path_expressed_by_tool_index += str(path[node_idx]["tool_index"])
                path_expressed_by_sign_of_tool_calling += str(path[node_idx]["post_sign"])
                
        
        time_stamp = extract_time_stamp_for_path(path)
        end_node = path[0]
        result_list = []
        
        if "label" in end_node:
            result_list.append(end_node["label"])
        else:
            for key in end_node:
                if key not in deleted_keys:
                    result_list.append(end_node[key])
                    
        simplified_paths.append({
            "path_expressed_by_tool_index": path_expressed_by_tool_index,
            "path_expressed_by_sign_of_tool_calling": path_expressed_by_sign_of_tool_calling,
            "result": result_list,
            "time_stamp": time_stamp
        })
        
    return simplified_paths

def find_path_recursively(results, map_from_tool_calling_id_to_tool_index, tool_calling_id_ending, output_item_idx):
    '''
    This function recursively finds the path of the results starting from each output item

    Args:
        results (dict): The results from the search query
        map_from_tool_calling_id_to_tool_index (dict): map from tool_calling id to tool index
        
    Returns:
        list: The execution path
    '''
    
    collected_paths = []
    tool_calling_ending = results[tool_calling_id_ending]
    tool_index_ending = map_from_tool_calling_id_to_tool_index[tool_calling_id_ending]
    
    tool_name = tool_calling_ending["tool_calling_information"]["tool_name"]
    post_sign_ending = tool_calling_ending["tool_calling_result"]["output"]["post_sign"][output_item_idx]
    
    # create the ending node of the path
    node_ending = {
        "tool_index": tool_index_ending,
        "tool_name": tool_name,
        "tool_calling_id": tool_calling_id_ending,
        "post_sign": post_sign_ending,
    }
    
    # add the output item into the node, except for pre_sign and post_sign
    for output_item_name in tool_calling_ending["tool_calling_result"]["output"]:
        if output_item_name != "pre_sign" and output_item_name != "post_sign":
            node_ending[output_item_name] = tool_calling_ending["tool_calling_result"]["output"][output_item_name][output_item_idx]
            
    
    # the function to find the path recursively
    def dfs_search(current_tool_calling_id, current_path, current_output_item_idx):
        # tool calling id is the id of 'initial input', it means the finding process is finished
        if current_tool_calling_id < 0:
            collected_paths.append(current_path)
            return
        
        pre_sign = results[current_tool_calling_id]["tool_calling_result"]["output"]["pre_sign"][current_output_item_idx]
        
        for dep_tool_calling_id_idx in range(len(results[current_tool_calling_id]["tool_calling_information"]["dep"])):
            dep_tool_calling_id = results[current_tool_calling_id]["tool_calling_information"]["dep"][dep_tool_calling_id_idx]
            
            if dep_tool_calling_id == -1:
                dep_tool_calling_id = -1 * (current_tool_calling_id + 1)
                
            dep_tool_calling_result = results[dep_tool_calling_id]["tool_calling_result"]
            dep_post_sign_list = dep_tool_calling_result["output"]["post_sign"]
            
            dep_output_item_idx = None
            for i in range(len(dep_post_sign_list)):
                if dep_post_sign_list[i] == pre_sign:
                    dep_output_item_idx = i
                    break
            
            assert dep_output_item_idx is not None, "Tool calling id: {}. Cannot find the corresponding post_sign for the pre_sign {}".format(current_tool_calling_id, pre_sign)
                
            node_newly_found = {
                "tool_index": map_from_tool_calling_id_to_tool_index[dep_tool_calling_id],
                "tool_name": results[dep_tool_calling_id]["tool_calling_information"]["tool_name"],
                "tool_calling_id": dep_tool_calling_id,
                "post_sign": pre_sign,
            }
            
            for output_item_name in dep_tool_calling_result["output"]:
                if output_item_name != "pre_sign" and output_item_name != "post_sign":
                    node_newly_found[output_item_name] = dep_tool_calling_result["output"][output_item_name][dep_output_item_idx]
                    
            # add the newly found node into the path and continue the search
            dfs_search(dep_tool_calling_id, current_path + [node_newly_found], dep_output_item_idx)
            
    # start the search
    dfs_search(tool_calling_id_ending, [node_ending], output_item_idx)
    
    return collected_paths

def find_paths(original_results):
    '''
    This function finds the execution paths of the results

    Args:
        original_results (dict): The original results from the search query
        
    Returns:
        list: The execution paths
    '''
    
    results = unify_tool_calling_ids_as_int(original_results)
    (
        _, 
        _, 
        map_from_tool_calling_id_to_tool_index_inference, 
        _, 
        tool_calling_ids_ending, 
        _
    ) = extract_mappings(results)
    
    execution_paths = []
    
    for tool_calling_id_ending in tool_calling_ids_ending:
        post_sign_list = results[tool_calling_id_ending]["tool_calling_result"]["output"]["post_sign"]
        
        # starting from each output item, from the last tool calling to the first tool calling
        for output_item_idx in range(len(post_sign_list)):
            newly_found_paths = find_path_recursively(results, map_from_tool_calling_id_to_tool_index_inference, tool_calling_id_ending, output_item_idx)
            execution_paths += newly_found_paths
            
    return simplify_paths(execution_paths)

def complete_info(single_path, tool_index_to_keep, map_from_tool_index_to_tool_name_inference, cut_paths):
    completed_result_info = {
        
    }
    existing_result_item = single_path["path_expressed_by_tool_index"].split("->")
    for tool_index in tool_index_to_keep:
        if tool_index not in existing_result_item:
            completed_result_info[map_from_tool_index_to_tool_name_inference[tool_index]] = None
    
    existing_result_item_signs = single_path["path_expressed_by_sign_of_tool_calling"].split("->")
    for i in range(len(existing_result_item)):
        signs_to_search = existing_result_item_signs[0:i+1]
        path_expressed_by_sign_to_search = "->".join(signs_to_search)
        result_found = None
        for path in cut_paths:
            if path["path_expressed_by_sign_of_tool_calling"] == path_expressed_by_sign_to_search:
                result_found = path
                break
        
        assert result_found is not None
        completed_result_info[map_from_tool_index_to_tool_name_inference[str(existing_result_item[i])]] = result_found
    
    
    return completed_result_info

def merge_results_by_obj(path_group_by_sign):
    merge_unit = ["object-detection-rgb", "object-detection-depth", "speaker-diarization-audio", "person-detection-rgb", "person-detection-depth"]
    merged_results_by_timestamp = {}
    
    # merge the results of the same object expressed by the unique sign
    for sign in path_group_by_sign:
        merged_result_info_of_sign = {}
        result_list_of_sign = path_group_by_sign[sign]
        
        for result_item in result_list_of_sign[0]:
            if result_item not in merge_unit:
                merged_result_info_of_sign[result_item] = None # if the result item is not in the merge unit, we just keep it
            else:
                if result_list_of_sign[0][result_item] is None:
                    merged_result_info_of_sign[result_item] = None
                    continue
                merged_result_info_of_sign[result_item] = result_list_of_sign[0][result_item]["result"] # if the result item is in the merge unit, we keep the first result item
        
        for i in range(0, len(result_list_of_sign)):
            result_info = result_list_of_sign[i]
            
            for result_item in result_info:
                if result_item in merge_unit or result_info[result_item] is None:
                    continue
                else:
                    if merged_result_info_of_sign[result_item] is not None:
                        if result_item == "face-detection-rgb" and set(result_info[result_item]).issubset(set(merged_result_info_of_sign[result_item])):
                            continue
                        elif result_item in ["face-recognition-rgb", "facial-expression-recognition-rgb"]:
                            for j in range(len(result_info[result_item]['result'])):
                                label = result_info[result_item]["result"][j]
                                # merged_result_info_of_sign[result_item].append(label + " (from {})".format(result_info["face-detection-rgb"]['result'][j]))
                                merged_result_info_of_sign[result_item].append(label)
                        else:
                            merged_result_info_of_sign[result_item] += result_info[result_item]["result"]
                    else:
                        if result_item in ["face-recognition-rgb", "facial-expression-recognition-rgb"]:
                            merged_result_info_of_sign[result_item] = []
                            for j in range(len(result_info[result_item]['result'])):
                                label = result_info[result_item]["result"][j]
                                # print("Result info: ", result_info)
                                # merged_result_info_of_sign[result_item].append(label + " (from {})".format(result_info["face-detection-rgb"]['result'][j]))
                                merged_result_info_of_sign[result_item].append(label)
                        else:
                            merged_result_info_of_sign[result_item] = result_info[result_item]["result"]
        
        timestamp = None
        for i in range(len(result_list_of_sign)):
            for result_item in result_list_of_sign[i]:
                if result_list_of_sign[i][result_item] is not None:
                    timestamp = result_list_of_sign[i][result_item]["time_stamp"]
                    break
        
        assert timestamp is not None, "The timestamp should not be None"
            
        if timestamp not in merged_results_by_timestamp:
            merged_results_by_timestamp[timestamp] = []
        merged_results_by_timestamp[timestamp].append(merged_result_info_of_sign)
    
    return merged_results_by_timestamp

def _get_path_modality(path, bi_map_from_tool_index_to_tool_name, modality_map):
    last_tool_index = path["path_expressed_by_tool_index"].split("->")[-1]
    tool_name = bi_map_from_tool_index_to_tool_name[last_tool_index]
    return modality_map[tool_name]

def _supplement_ending_tools(tool_index_by_modality,
                             original_tool_calling_ids_ending,
                             original_map_from_tool_calling_id_to_tool_index_inference,
                             bi_map_from_tool_index_to_tool_name,
                             modality_map):
    """
    补充属于各模态但未在路径中出现的结束节点工具
    
    Args:
        tool_index_by_modality: 按模态分组的工具索引集合（会被修改）
        original_tool_calling_ids_ending: 原始结束节点ID列表
        original_map_from_tool_calling_id_to_tool_index_inference: 工具调用ID到索引的映射
        bi_map_from_tool_index_to_tool_name: 工具索引到名称的双向映射
        modality_map: 工具名称到模态的映射
    """
    for tool_calling_id_ending in original_tool_calling_ids_ending:
        tool_index_ending = original_map_from_tool_calling_id_to_tool_index_inference[tool_calling_id_ending]
        tool_name_ending = bi_map_from_tool_index_to_tool_name[tool_index_ending]
        modality = modality_map[tool_name_ending]
        
        if modality in tool_index_by_modality:
            if tool_index_ending not in tool_index_by_modality[modality]:
                tool_index_by_modality[modality].add(tool_index_ending)

def group_paths_and_tools_by_modality(paths_without_sub_path,
                                       original_tool_calling_ids_ending,
                                       original_map_from_tool_calling_id_to_tool_index_inference,
                                       bi_map_from_tool_index_to_tool_name):
    """
    按模态对路径进行分组，并收集每个模态涉及的工具索引
    
    Args:
        paths_without_sub_path: 去除子路径后的路径列表
        original_tool_calling_ids_ending: 原始结束节点ID列表
        original_map_from_tool_calling_id_to_tool_index_inference: 工具调用ID到索引的映射
        bi_map_from_tool_index_to_tool_name: 工具索引到名称的双向映射
        
    Returns:
        tuple: (path_group_by_modality, tool_index_by_modality)
    """
    path_group_by_modality = {}
    tool_index_by_modality = {}
    modality_map = get_map_from_tool_name_to_modality()
    
    for path in paths_without_sub_path:
        if path["path_expressed_by_tool_index"] == "":
            continue
        
        # 获取路径的模态（基于最后一个工具）
        modality = _get_path_modality(path, bi_map_from_tool_index_to_tool_name, modality_map)
        
        # 将路径添加到对应模态组
        if modality not in path_group_by_modality:
            path_group_by_modality[modality] = []
        path_group_by_modality[modality].append(path)
        
        # 收集该路径涉及的所有工具索引
        tool_indexes = path["path_expressed_by_tool_index"].split("->")
        if modality not in tool_index_by_modality:
            tool_index_by_modality[modality] = set()
        tool_index_by_modality[modality].update(tool_indexes)
    
    _supplement_ending_tools(
        tool_index_by_modality,
        original_tool_calling_ids_ending,
        original_map_from_tool_calling_id_to_tool_index_inference,
        bi_map_from_tool_index_to_tool_name,
        modality_map
    )
    
    tool_index_by_modality = {
        modality: list(tool_set) 
        for modality, tool_set in tool_index_by_modality.items()
    }
    
    return path_group_by_modality, tool_index_by_modality

def _group_paths_by_first_tool_sign(paths):
    """
    按第一个工具调用签名对路径进行分组
    
    Args:
        paths: 路径列表
        
    Returns:
        dict: 按第一个工具签名分组的路径
    """
    path_group_by_first_sign = {}
    
    for path in paths:
        first_tool_calling_sign = path["path_expressed_by_sign_of_tool_calling"].split("->")[0]
        
        if first_tool_calling_sign not in path_group_by_first_sign:
            path_group_by_first_sign[first_tool_calling_sign] = []
        path_group_by_first_sign[first_tool_calling_sign].append(path)
    
    return path_group_by_first_sign

def _complete_path_groups(path_group_by_first_sign,
                          tool_indexes,
                          bi_map_from_tool_index_to_tool_name,
                          cut_paths):
    """
    完成每组路径的信息处理
    
    Args:
        path_group_by_first_sign: 按第一个工具签名分组的路径
        tool_indexes: 该模态涉及的所有工具索引
        bi_map_from_tool_index_to_tool_name: 工具索引到名称的双向映射
        cut_paths: 裁剪后的路径列表
        
    Returns:
        dict: 完成信息后的路径组
    """
    completed_groups = {}
    
    for first_tool_calling_sign, paths in path_group_by_first_sign.items():
        completed_result_list = []
        
        for path in paths:
            completed_result = complete_info(
                path,
                tool_indexes,
                bi_map_from_tool_index_to_tool_name,
                cut_paths
            )
            completed_result_list.append(completed_result)
        
        completed_groups[first_tool_calling_sign] = completed_result_list
    
    return completed_groups

def process_modality_results(path_group_by_modality,
                             tool_index_by_modality,
                             bi_map_from_tool_index_to_tool_name,
                             cut_paths):
    """
    处理每个模态的路径，生成最终结果
    
    Args:
        path_group_by_modality: 按模态分组的路径
        tool_index_by_modality: 按模态分组的工具索引
        bi_map_from_tool_index_to_tool_name: 工具索引到名称的双向映射
        cut_paths: 裁剪后的路径列表
        
    Returns:
        dict: 按模态分组的处理结果
    """
    result_group_by_modality = {}
    
    for modality, single_modality_paths in path_group_by_modality.items():
       
        path_group_by_first_sign = _group_paths_by_first_tool_sign(single_modality_paths)
        
        
        completed_groups = _complete_path_groups(
            path_group_by_first_sign,
            tool_index_by_modality[modality],
            bi_map_from_tool_index_to_tool_name,
            cut_paths
        )
        
        result_group_by_modality[modality] = copy.deepcopy(
            merge_results_by_obj(completed_groups)
        )
    
    return result_group_by_modality

def process_paths_by_modality(paths_without_sub_path, 
                              original_tool_calling_ids_ending,
                              original_map_from_tool_calling_id_to_tool_index_inference,
                              bi_map_from_tool_index_to_tool_name,
                              cut_paths):
    """
    按模态处理路径并生成最终结果
    
    Args:
        paths_without_sub_path: 去除子路径后的路径列表
        original_tool_calling_ids_ending: 原始结束节点ID列表
        original_map_from_tool_calling_id_to_tool_index_inference: 工具调用ID到索引的映射
        bi_map_from_tool_index_to_tool_name: 工具索引到名称的双向映射
        cut_paths: 裁剪后的路径列表
        
    Returns:
        dict: 按模态分组的结果
    """
    
    path_group_by_modality, tool_index_by_modality = group_paths_and_tools_by_modality(
        paths_without_sub_path,
        original_tool_calling_ids_ending,
        original_map_from_tool_calling_id_to_tool_index_inference,
        bi_map_from_tool_index_to_tool_name
    )
    
    
    result_group_by_modality = process_modality_results(
        path_group_by_modality,
        tool_index_by_modality,
        bi_map_from_tool_index_to_tool_name,
        cut_paths
    )
    
    
    result_group_by_modality = json.loads(json.dumps(result_group_by_modality, sort_keys=True))
    
    return result_group_by_modality

def format_results_to_dict(original_results):
    '''
    This function formats the results of the search query

    Args:
        original_results (dict): The original results from the search query
        
    Returns:
        dict: The formatted results
    '''
    all_paths = []
    
    original_results = adjust_dep_ids(original_results)
    tool_name_to_delete = ["initial input"] + [tool["tool_name"] for tool in get_all_vocabulary_set_for_tool_type("db_tool")]
    
    bi_map_from_tool_index_to_tool_name, _ , original_map_from_tool_calling_id_to_tool_index_inference, _, original_tool_calling_ids_ending, _ = extract_mappings(original_results)
    tool_index_to_delete = [bi_map_from_tool_index_to_tool_name.inv[tool_name] for tool_name in tool_name_to_delete]
    
    while len(original_results) > 0:
        execution_paths = find_paths(original_results)
        _, _, _, _, tool_calling_ids_ending, _ = extract_mappings(original_results)
        
        all_paths += execution_paths
        remove_result_by_key(original_results, tool_calling_ids_ending)
    
    # cut the path by removing the tools to delete
    cut_paths = []
    for path in all_paths:
        cut_paths.append(cut_path(path, tool_index_to_delete))
    
    # remove the sub paths
    paths_without_sub_path = remove_subpath(cut_paths)
    
    # process the paths by modality
    result_group_by_modality = process_paths_by_modality(
        paths_without_sub_path,
        original_tool_calling_ids_ending,
        original_map_from_tool_calling_id_to_tool_index_inference,
        bi_map_from_tool_index_to_tool_name,
        cut_paths
    )
    
    
    return result_group_by_modality

def find_interupt(formatted_results):
    '''
    Check if the results contain any execution paths that are interrupted

    Args:
        results (dict): the execution results
        
    Returns:
        bool: True if the results contain any interrupted execution paths
    '''
    contain_interrupt = False
    
    for modality in formatted_results:
        for time_interval in formatted_results[modality]:
            for each_result in formatted_results[modality][time_interval]:
                for result_item_key in each_result:
                    if each_result[result_item_key] is None:
                        contain_interrupt = True
                        break
                    
    return contain_interrupt

def aggregate_results(results_of_each_time_range):
    aggregated_results = {}
    
    for time_range_str in results_of_each_time_range:
        # assert len(results_of_each_time_range[time_range_str]) == 1, "The result of each time range should only contain one modality"
        
        for i in range(0, len(results_of_each_time_range[time_range_str])):
            result_of_current_time_range = results_of_each_time_range[time_range_str][i]
            modality = list(result_of_current_time_range.keys())[0]
            
            if modality not in aggregated_results:
                aggregated_results[modality] = {}
                
            for time_interval in result_of_current_time_range[modality]:
                if time_interval not in aggregated_results[modality]:
                    aggregated_results[modality][time_interval] = []
                    
                aggregated_results[modality][time_interval] += result_of_current_time_range[modality][time_interval]
            
    for modality in aggregated_results:
        for time_interval in aggregated_results[modality]:
            for i in range(len(aggregated_results[modality][time_interval])):
                if "face-detection-rgb" in aggregated_results[modality][time_interval][i] and aggregated_results[modality][time_interval][i]["face-detection-rgb"] is not None:
                    # delete the "face-detection-rgb" key
                    aggregated_results[modality][time_interval][i].pop("face-detection-rgb")
                if "face-recognition-rgb" in aggregated_results[modality][time_interval][i] and aggregated_results[modality][time_interval][i]["face-recognition-rgb"] is not None:
                    # voting
                    aggregated_results[modality][time_interval][i]["face-recognition-rgb"] = Counter(aggregated_results[modality][time_interval][i]["face-recognition-rgb"]).most_common(1)[0][0]
                if "facial-expression-recognition-rgb" in aggregated_results[modality][time_interval][i] and aggregated_results[modality][time_interval][i]["facial-expression-recognition-rgb"] is not None:
                    # voting
                    aggregated_results[modality][time_interval][i]["facial-expression-recognition-rgb"] = Counter(aggregated_results[modality][time_interval][i]["facial-expression-recognition-rgb"]).most_common(1)[0][0]
                
    return aggregated_results













import json
import os

import yaml



# initialize the vocabulary set
def initialize_vocabulary_set():
    '''
    This function initializes the vocabulary set.
    '''
    
    # read yaml file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    vocabulary_set_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "vocabulary_set.json")

    # create an empty vocabulary set
    vocabulary_set = {
        "model_tool": [
            {
                "tool_name": "object-detection-rgb",
                "description": "This tool will detect objects in the input video and crop the detected objects into seperate videos.",
                "input": {"rgb_video": "The input video to be processed."},
                "output": {
                    "label": "A list of list. Each list contains all the labels of the detected objects in the corresponding video. Possible labels: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']",
                    "rgb_video": "The cropped videos of the detected objects.",
                },
                "modality": "rgb",
            },
            {
                "tool_name": "human-activity-classification-rgb",
                "description": "This tool will recognize the human activity in each input video.",
                "input": {"rgb_video": "The input videos of person to be processed."},
                "output": {
                    "label": "The list of label of the recognized human activity in each video. Possible labels: ['Boxing', 'Pushing', 'Patting on the shoulder', 'Shaking hands', 'Face-to-face conversation', 'Standing still', 'Waving arms', 'Playing with phone using both hands', 'Rubbing hands', 'Touching head', 'Coughing', 'Jumping with both feet', 'Stomping feet', 'Walking', 'Squatting', 'Bending over', 'Throwing things', 'Picking up things', 'Kicking the cabinet', 'Searching for things in the cabinet', 'Angrily hitting the cabinet', 'Sitting still', 'Sitting and making a phone call', 'Sitting down', 'Standing up', 'Raising hand to check time/putting down', 'Stretching', 'Taking off/putting on a mask', 'Lying on a mat', 'Falling down/getting up']"
                },
                "modality": "rgb",
            },
            {
                "tool_name": "object-detection-depth",
                "description": "This tool will detect objects in the input depth video and crop the detected objects into seperate videos.",
                "input": {"depth_video": "The input depth video to be processed."},
                "output": {
                    "label": "A list of list. Each list contains all the labels of the detected objects in the corresponding video. Possible labels: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']",
                    "depth_video": "The cropped videos of the detected objects.",
                },
                "modality": "depth",
            },
            {
                "tool_name": "human-activity-classification-depth",
                "description": "This tool will recognize the human activity in each input depth video.",
                "input": {"depth_video": "The input depth videos of person to be processed."},
                "output": {
                    "label": "The list of label of the recognized human activity in each depth video. Possible labels: ['Boxing', 'Pushing', 'Patting on the shoulder', 'Shaking hands', 'Face-to-face conversation', 'Standing still', 'Waving arms', 'Playing with phone using both hands', 'Rubbing hands', 'Touching head', 'Coughing', 'Jumping with both feet', 'Stomping feet', 'Walking', 'Squatting', 'Bending over', 'Throwing things', 'Picking up things', 'Kicking the cabinet', 'Searching for things in the cabinet', 'Angrily hitting the cabinet', 'Sitting still', 'Sitting and making a phone call', 'Sitting down', 'Standing up', 'Raising hand to check time/putting down', 'Stretching', 'Taking off/putting on a mask', 'Lying on a mat', 'Falling down/getting up']"
                },
                "modality": "depth",
            },
        ],
        "db_tool": [
            {
                "tool_name": "query-video-data-rgb",
                "description": "The query-video-data operation is used to retrieve or fetch RGB video data from a database based on inputted time range.",
                "input": {"text": "The time range of the video data to be retrieved."},
                "output": {"rgb_video": "The retrieved video data."},
                "modality": "rgb",
            },
            {
                "tool_name": "query-video-data-depth",
                "description": "The query-video-data-depth operation is used to retrieve or fetch depth video data from a database based on inputted time range.",
                "input": {
                    "text": "The time range of the depth video data to be retrieved."
                },
                "output": {"depth_video": "The retrieved depth video data."},
                "modality": "depth",
            },
        ],
        "sensor_tool": [
            {
                "tool_name": "take-picture",
                "description": "Use the camera to take a specified number of pictures and store them in the database. ",
            },
            {
                "tool_name": "send-notification",
                "description": "Send a notification to the user's mobile phone.",
                "input": {"text": "The content of the notification."},
                "output": {"status": "The result of sending the notification."},
            },
        ],
    }

    # write the empty vocabulary set to the file
    with open(vocabulary_set_path, "w") as f:
        json.dump(vocabulary_set, f, indent=4)

    print("Vocabulary set initialized successfully")

# Register a new tool in the vocabulary set
def register_tool(
    tool_type, tool_name, tool_description, input_args, output_items, modality
):
    '''
    This function registers a new tool in the vocabulary set.

    Args:
        tool_type (str): the category of the tool
        tool_name (str): the name of the tool
        tool_description (str): the description of the tool's functionality
        input_args (dict): the input arguments of the tool
        output_items (dict): the output items of the tool
        modality (dict): the modality of the tool. If the tool does not have a modality, set it to None.
    '''
    
    # read yaml file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    vocabulary_set_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "vocabulary_set.json")

    # construct the tool information json object
    tool = {
        "tool_name": tool_name,
        "tool_description": tool_description,
        "input": input_args,
        "output": output_items,
    }

    # add modality if it is not None
    if modality is not None:
        tool["modality"] = modality

    # read the vocabulary set from the file
    with open(vocabulary_set_path, "r") as f:
        vocabulary_set = json.load(f)

    # if the tool type already exists in the vocabulary set, append the tool to the list
    if tool_type in vocabulary_set:
        for existing_tool in vocabulary_set[tool_type]:
            if existing_tool["tool_name"] == tool_name:
                print("Tool already exists in the vocabulary set")
                return

        vocabulary_set[tool_type].append(tool)
    # if the tool type does not exist in the vocabulary set, create a new entry
    else:
        vocabulary_set[tool_type] = [tool]

    # write the updated vocabulary set back to the file
    with open(vocabulary_set_path, "w") as f:
        json.dump(vocabulary_set, f, indent=4)

    print("Tool registered successfully")

# Remove a tool from the vocabulary set
def remove_tool(tool_name):
    '''
    This function removes a tool from the vocabulary set.

    Args:
        tool_name (str): the name of the tool to be removed
    '''
    
    # read yaml file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    vocabulary_set_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "vocabulary_set.json")

    # read the vocabulary set from the file
    with open(vocabulary_set_path, "r") as f:
        vocabulary_set = json.load(f)

    # iterate through the vocabulary set to find the tool to be removed
    for tool_type in vocabulary_set:
        # record the original length of the tool list
        original_length = len(vocabulary_set[tool_type])
        # onli keep the tools that are not the one to be removed
        vocabulary_set[tool_type] = [
            tool for tool in vocabulary_set[tool_type] if tool["tool_name"] != tool_name
        ]
        # record the updated length of the tool list
        updated_length = len(vocabulary_set[tool_type])
        # if the two lengths are the same, the tool to be removed does not exist
        if original_length == updated_length:
            print("Tool does not exist in the vocabulary set")
            return

    # write the updated vocabulary set back to the file
    with open(vocabulary_set_path, "w") as f:
        json.dump(vocabulary_set, f, indent=4)

    print("Tool removed successfully")

# get the entire vocabulary set
def get_all_vocabulary_set():
    '''
    This function returns the entire vocabulary set as a json object.

    Returns:
        dict: the entire vocabulary set
    '''
    
    # read yaml file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    vocabulary_set_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "vocabulary_set.json")

    # read the vocabulary set from the file
    with open(vocabulary_set_path, "r") as f:
        vocabulary_set = json.load(f)

    return vocabulary_set

# get all vocabulary set for a specific tool type
def get_all_vocabulary_set_for_tool_type(tool_type):
    '''
    This function returns all the tools of a specific type in the vocabulary set.

    Returns:
        list: the list of tools of the specified type
    '''
    
    # read yaml file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    vocabulary_set_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "vocabulary_set.json")

    # read the vocabulary set from the file
    with open(vocabulary_set_path, "r") as f:
        vocabulary_set = json.load(f)

    return vocabulary_set[tool_type]

def get_all_modalities():
    '''
    This function returns all the modalities in the vocabulary set.

    Returns:
        list: the list of modalities
    '''
    
    # read yaml file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    modalities_list = config["execution_settings"]["modalities"][dataset_name]
    
    return modalities_list

def get_map_from_tool_name_to_modality():
    '''
    This function returns a map from tool name to modality.

    Returns:
        dict: the map from tool name to modality
    '''
    
    # read yaml file
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    dataset_name = config["dataset_name"]
    vocabulary_set_path = os.path.join(config["planning_settings"]["tool_files_path"], dataset_name, "vocabulary_set.json")

    # read the vocabulary set from the file
    with open(vocabulary_set_path, "r") as f:
        vocabulary_set = json.load(f)

    tool_name_to_modality = {}

    for tool_type in vocabulary_set:
        for tool in vocabulary_set[tool_type]:
            tool_name_to_modality[tool["tool_name"]] = tool.get("modality", None)

    return tool_name_to_modality

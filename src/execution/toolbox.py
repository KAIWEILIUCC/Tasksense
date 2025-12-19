import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import numpy as np
import torch
import yaml
from deep_sort_realtime.deepsort_tracker import DeepSort
from thop import profile
from ultralytics import YOLO

from src.execution.model_zoo.har_model import TSN
from src.utils import *
from src.utils import (MyVideoCapture, ObjectDetectionVideoCapture,
                       get_video_info, load_cache_with_lock,
                       save_cache_with_lock,
                       save_flops_cache_with_lock)
from sshtunnel import SSHTunnelForwarder
import paramiko

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="deep_sort_realtime")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# read the configuration file
config_path = os.environ["CONFIG_PATH"]
config = yaml.load(open(config_path), Loader=yaml.FullLoader)
dataset_name = config["dataset_name"]
intermediate_output_path = config["execution_settings"]["intermediate_output_path"][dataset_name]
device = torch.device(config["execution_settings"]["device"])
use_cache = config["execution_settings"]["use_cache"]
execution_cache_path = config["execution_settings"]["execution_cache_path"]
flops_cache_path = config["execution_settings"]["flops_cache_path"]
mearue_flops = config["execution_settings"]["measure_flops"]


# 1. model tools
# (1) RGB modality
# model tool: object-detection-rgb
def model_tool_object_detection_rgb(
    args,
    time_of_each_stage_of_tool_execution,
    device=device,
    specified_saving_path=None,
    gflops_of_tool=None
):
    '''
    This function is the implementation of the object-detection-rgb tool.

    Args:
        args (dict): The arguments of the tool.
        time_of_each_stage_of_tool_execution (dict): The time of each stage of the tool execution.
        device (torch.device, optional): The device to use. Defaults to device.
        specified_saving_path (str, optional): It specifies the path to save the output of the tool. Defaults to None.

    Returns:
        dict: The result of the tool execution.
    '''
    if not args["execute_next"]:
        print_tips("Object-detection-rgb skips.", emoji="zap", text_color="yellow", border=False)
        return {
            "task": "object-detection-rgb",
            "output": {
                "label": [],
                "pre_sign": [],
                "post_sign": [],
            },
            "execute_next": False
        }
    
    print_tips("Object-detection-rgb Task Starts.", emoji="zap", text_color="yellow", border=False)
    
    saving_folder = config["execution_settings"]["tool_name_saving_folder_map"][dataset_name]["object-detection-rgb"]
    
    # check if the specified_saving_path is given
    # if it is not given, then save the output in the default path
    # otherwise, save the output in the specified path
    if specified_saving_path is None:
        saving_path = os.path.join(intermediate_output_path, saving_folder)
    else:
        saving_path = specified_saving_path
        
    os.makedirs(intermediate_output_path, exist_ok=True)
    os.makedirs(saving_path, exist_ok=True)
    
    # pre-process the input
    start_time_preprocessing = time.time()
    video_paths = args["output"]["rgb_video"] # the list of the paths of the input videos
    pre_signs = args["output"]["post_sign"] # the signs of the previous tool calls' outputs
    end_time_preprocessing = time.time()
    
    # load the model
    start_time_model_loading = time.time()
    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=30, embedder="mobilenet")
    end_time_model_loading = time.time()
    
    # inference
    start_time_inference = time.perf_counter()
    
    result_video_paths = [] # list to store the output cropped videos' paths
    result_label = [] # list to store the output labels
    result_pre_sign = [] # list to store the signs of the previous tool calls' outputs
    result_post_sign = [] # list to store the signs of the output of this tool call
    batch_size = 64 # batch size for the inference
    
    # if use_cache is True, then load the cache
    if use_cache:
        cache = load_cache_with_lock(execution_cache_path)
        
    if mearue_flops:
        gflops = 0.0
        flops_cache = load_cache_with_lock(flops_cache_path)
    
    for video_path_idx in range(len(video_paths)):
        video_path = video_paths[video_path_idx]
        
        # if use_cache is True and the video path is in the cache
        # then skip the video and directly use the results in the cache
        if use_cache:
            if video_path in cache["object-detection-rgb"]:
                print_tips(f"[object-detection-rgb] Video {video_path} has been processed before. Skip.", emoji="zap", text_color="red", border=False)
                result_video_paths.extend(cache["object-detection-rgb"][video_path]["rgb_video"])
                result_label.extend(cache["object-detection-rgb"][video_path]["label"])
                
                # create the pre_sign and post_sign for the skipped video
                # and append them to the result dictionary
                for _ in cache["object-detection-rgb"][video_path]["rgb_video"]:
                    result_pre_sign.append(pre_signs[video_path_idx])
                    result_post_sign.append(generate_a_sign())
                    
                continue
        
        result_video_paths_of_this_video = [] # list to store the output cropped videos' paths of this video
        result_label_of_this_video = [] # list to store the output labels of this video
        map_from_track_id_to_cv2_writer = {} # map from track id to cv2 writer
        map_from_track_id_to_cropped_video_path = {} # map from track id to cropped video path
        
        # extract the time information from the video path
        date = video_path.split("/")[-2]
        time_point = video_path.split("/")[-1].split(".")[0]
        
        # read the video and get the information of the video
        frame_width, frame_height, fps, frame_count = get_video_info(video_path)
        
        print_tips(f"[object-detection-rgb] Video Information: Frame Width: {frame_width}, Frame Height: {frame_height}, FPS: {fps}, Frame Count: {frame_count}, Path: {video_path}", emoji="zap", text_color="yellow", border=False)
        
        if any([frame_width == 0, frame_height == 0, fps == 0, frame_count == 0]):
            continue
        
        # clear all the tracks
        tracker.delete_all_tracks()
        
        # start to process the video
        # cap = ObjectDetectionVideoCapture(video_path)
        cap = MyVideoCapture(video_path)
        processed_frame_count = 0
        while not cap.end:
            _, _ = cap.read()
            # if the stack is full or the video ends
            if (len(cap.stack) == batch_size) or (cap.end and len(cap.stack) != 0):
                _, frame_list = cap.get_video_clip()
                processed_frame_count += len(frame_list)
                
                sys.stdout.write(f"\r[object-detection-rgb] Processed Frames: {processed_frame_count}/{frame_count} = {processed_frame_count/frame_count:.2f}")
                sys.stdout.flush()
                
                with torch.no_grad():
                    detection_list = model(frame_list, verbose=False)
                    
                    if mearue_flops:
                        calcutae_device = torch.device("cpu")
                        
                        # YOLO
                        flops_key_yolo = f"object-detection-rgb-yolo-{(len(frame_list), 3, 256, 256)}"
                        
                        if flops_key_yolo in flops_cache:
                            flops_yolo = flops_cache[flops_key_yolo]
                        else:
                            yolo_model = model.model
                            yolo_model.eval()
                            input_data = torch.randn(len(frame_list), 3, 256, 256)
                            yolo_model = yolo_model.to(calcutae_device)
                            input_data = input_data.to(calcutae_device)
                            flops_yolo, _ = profile(yolo_model, inputs=(input_data, ), verbose=False)
                            flops_yolo = flops_yolo / 1e9
                            
                            # save the FLOPs to the cache
                            flops_cache[flops_key_yolo] = flops_yolo
                            save_flops_cache_with_lock(flops_cache_path, flops_cache, flops_key_yolo)
                            
                            yolo_model = yolo_model.to(device)
                            
                        gflops += flops_yolo
        
                        # DeepSort
                        flops_key_deepsort = f"object-detection-rgb-tracker-{(len(frame_list), 3, 224, 224)}"
                        if flops_key_deepsort in flops_cache:
                            flops_tracker = flops_cache[flops_key_deepsort]
                        else:
                            tracker_model = tracker.embedder.model
                            tracker_model.eval()
                            # input should be torch.HalfTensor
                            input_data_embedder = torch.randn(len(frame_list), 3, 224, 224).half()
                            tracker_model = tracker_model.to(calcutae_device)
                            input_data_embedder = input_data_embedder.to(calcutae_device)
                            flops_tracker, _ = profile(tracker_model, inputs=(input_data_embedder, ), verbose=False)
                            flops_tracker = flops_tracker / 1e9
                            
                            # save the FLOPs to the cache
                            flops_cache[flops_key_deepsort] = flops_tracker
                            save_flops_cache_with_lock(flops_cache_path, flops_cache, flops_key_deepsort)
                            
                            tracker_model = tracker_model.to(device)
                            
                        gflops += flops_tracker
                        # =====================================
                
                formatted_detections = []
                class_names = detection_list[0].names
                
                for detection in detection_list:
                    boxes = detection.boxes
                    formatted_detections.append({
                        "boxes": boxes.xyxy,
                        "scores": boxes.conf,
                        "labels": boxes.cls,
                    })
                    
                # convert the detection to the format required by the tracker
                # in this case, we only focus on the person class
                focused_classes = [0]
                convert_detection_partial = partial(
                    convert_detection_object_detection, 
                    threshold=0.80, 
                    classes=focused_classes)
                with ThreadPoolExecutor() as executor:
                    detection_list = list(executor.map(
                        convert_detection_partial, 
                        formatted_detections))
                                        
                for i in range(len(detection_list)):
                    detection = detection_list[i]
                    frame = frame_list[i]
                    
                    tracks = tracker.update_tracks(detection, frame=frame)
                    
                    for track in tracks:
                        x1, y1, x2, y2 = track.to_tlbr()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        track_id = track.track_id
                        
                        # if a new track is found
                        if track_id not in map_from_track_id_to_cv2_writer:
                            saving_name = f"{date}_{time_point}_object_{track_id}_{class_names[int(track.det_class)]}.avi"
                            obj_video_path = os.path.join(
                                saving_path,
                                saving_name
                            )
                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            
                            # create the cv2 writer
                            # and put it in the map
                            map_from_track_id_to_cv2_writer[track_id] = cv2.VideoWriter(
                                obj_video_path,
                                fourcc,
                                fps,
                                (640, 480)
                            )
                            # put the video path in the map
                            map_from_track_id_to_cropped_video_path[track_id] = obj_video_path
                            
                            # obtain the pre_sign and post_sign
                            pre_sign = pre_signs[video_path_idx]
                            post_sign = generate_a_sign()
                            
                            # append the new results
                            result_video_paths.append(obj_video_path)
                            result_label.append(class_names[int(track.det_class)])
                            result_pre_sign.append(pre_sign)
                            result_post_sign.append(post_sign)
                            result_video_paths_of_this_video.append(obj_video_path)
                            result_label_of_this_video.append(class_names[int(track.det_class)])
                            
                        # crop the object frame and set the output height and width
                        obj_video_frame = frame[y1:y2, x1:x2]
                        output_height, output_width = 480, 640
                        
                        # if the object frame is not empty
                        # then write the object frame to the cv2 writer
                        if obj_video_frame.size != 0:
                            height_of_obj_video_frame, width_of_obj_video_frame = obj_video_frame.shape[:2]
                            scale = min(output_height / height_of_obj_video_frame, output_width / width_of_obj_video_frame)
                            
                            scaled_width = int(width_of_obj_video_frame * scale)
                            scaled_height = int(height_of_obj_video_frame * scale)
                            scaled_obj_video_frame = cv2.resize(obj_video_frame, (scaled_width, scaled_height))
                            
                            output_scaled_obj_video_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                            
                            x_offset = (output_width - scaled_width) // 2
                            y_offset = (output_height - scaled_height) // 2
                            
                            output_scaled_obj_video_frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = scaled_obj_video_frame
                            map_from_track_id_to_cv2_writer[track_id].write(output_scaled_obj_video_frame)
        
        
        if use_cache:
            cache["object-detection-rgb"][video_path] = {
                "rgb_video": result_video_paths_of_this_video,
                "label": result_label_of_this_video
            }
            save_cache_with_lock(execution_cache_path, cache, "object-detection-rgb", video_path)
                 
        cap.release()
        for writer in map_from_track_id_to_cv2_writer.values():
            writer.release()
            
    end_time_inference = time.perf_counter()
                            
    print_tips("Object-detection-rgb Task Ends.", emoji="tada", text_color="yellow", border=False)
    
    # record the time of each stage of the tool execution
    time_of_each_stage_of_tool_execution["pre_processing_time"] = end_time_preprocessing - start_time_preprocessing
    time_of_each_stage_of_tool_execution["model_loading_time"] = end_time_model_loading - start_time_model_loading
    time_of_each_stage_of_tool_execution["model_inference_time"] = end_time_inference - start_time_inference
    
    # record the FLOPs of the tool
    if mearue_flops:
        gflops_of_tool["gflops"] = gflops
    
    # construct the results
    results = {
        "tool_name": "object-detection-rgb",
        "output": {
            "rgb_video": result_video_paths,
            "label": result_label,
            "pre_sign": result_pre_sign,
            "post_sign": result_post_sign
        },
        "saving": True,
        "execute_next": True
    }
    
    # if there is no output then set execute_next to False
    if len(result_video_paths) == 0:
        results["execute_next"] = False
    
    return results

# model tool: human-activity-classification-rgb
def model_tool_human_activity_classification_rgb(
    args,
    time_of_each_stage_of_tool_execution,
    device=device,
    gflops_of_tool=None
):
    '''
    This function is the implementation of the human-activity-classification-rgb tool.

    Args:
        args (dict): The arguments of the tool.
        time_of_each_stage_of_tool_execution (dict): The time of each stage of the tool execution.
        device (torch.device, optional): The device to use. Defaults to device.

    Returns:
        dict: The result of the tool execution.
    '''
    if not args["execute_next"]:
        print_tips("Human-activity-classification-rgb skips.", emoji="zap", text_color="yellow", border=False)
        return {
            "task": "human-activity-classification-rgb",
            "output": {
                "label": [],
                "pre_sign": [],
                "post_sign": [],
            },
            "execute_next": False
        }
    
    print_tips("Human-activity-classification-rgb Task Starts.", emoji="zap", text_color="yellow", border=False)
    
    start_time_preprocessing = time.perf_counter()
    video_paths = args["output"]["rgb_video"]
    pre_signs = args["output"]["post_sign"]
    end_time_preprocessing = time.perf_counter()
    
    start_time_model_loading = time.perf_counter()
    model = TSN(num_class=30, dropout=0.8, modality='RGB', is_shift=True).eval()
    model.load_state_dict(torch.load(config["execution_settings"]["model_checkpoint_paths"][dataset_name]["human_activity_classification_rgb"]))
    model = model.to(device)
    
    label_map = {
        "a01": "boxing",
        "a02": "push",
        "a03": "pat on shoulder",
        "a04": "handshake",
        "a05": "face-to-face conversation",
        "a06": "standing still",
        "a07": "waving arms",
        "a08": "playing with phone using both hands",
        "a09": "rubbing hands",
        "a10": "touching head",
        "a11": "coughing",
        "a12": "jumping with both feet",
        "a13": "stomping feet",
        "a14": "walking",
        "a15": "squatting down",
        "a16": "bending over",
        "a17": "throwing things",
        "a18": "picking up things",
        "a19": "kicking cabinet",
        "a20": "looking for something in the cabinet",
        "a21": "angrily slapping the cabinet",
        "a22": "sitting still",
        "a23": "sitting and making a phone call",
        "a24": "standing up",
        "a25": "sitting down",
        "a26": "raising hand to check time/putting down",
        "a27": "stretching",
        "a28": "taking off mask/putting on mask",
        "a29": "lying on a mat",
        "a30": "falling down/getting up"
    }
    end_time_model_loading = time.perf_counter()
    
    start_time_inference = time.perf_counter()
    result_label = []
    result_pre_sign = []
    result_post_sign = []
    
    # if use_cache is True, then load the cache
    if use_cache:
        cache = load_cache_with_lock(execution_cache_path)
        
    if mearue_flops:
        gflops = 0.0
        flops_cache = load_cache_with_lock(flops_cache_path)
    
    for video_path_idx in range(len(video_paths)):
        video_path = video_paths[video_path_idx]
        
        # if use_cache is True and the video path is in the cache
        # then skip the video and directly use the results in the cache
        if use_cache:
            if video_path in cache["human-activity-classification-rgb"]:
                print_tips(f"[human-activity-classification-rgb] Video {video_path} has been processed before. Skip.", emoji="zap", text_color="yellow", border=False)
                result_label.extend(cache["human-activity-classification-rgb"][video_path]["label"])
                
                # create the pre_sign and post_sign for the skipped video
                # and append them to the result dictionary
                for _ in cache["human-activity-classification-rgb"][video_path]["label"]:
                    result_pre_sign.append(pre_signs[video_path_idx])
                    result_post_sign.append(generate_a_sign())
                
                continue
        
        label_of_each_set_of_frames = []
        label_of_this_video = []
        result_label_of_this_video = []
        
        # read the video and get the information of the video
        frame_width, frame_height, fps, frame_count = get_video_info(video_path)
        
        # if the total frame count is less than 8, then skip the video
        if frame_count < 8:
            continue
        
        # print(f"[human-activity-classification-rgb] Video Information: Frame Width: {frame_width}, Frame Height: {frame_height}, FPS: {fps}, Frame Count: {frame_count}, Path: {video_path}")
        print_tips(f"[human-activity-classification-rgb] Video Information: Frame Width: {frame_width}, Frame Height: {frame_height}, FPS: {fps}, Frame Count: {frame_count}, Path: {video_path}", emoji="zap", text_color="yellow", border=False)
        
        cap = HARVideoCapture(video_path)
        batch_size = 32
        
        while not cap.end:
            _, _ = cap.read()
            # if the stack is full or the video ends
            if len(cap.stack) == batch_size or (cap.end and len(cap.stack) != 0):
                clip, stack = cap.get_video_clip()
                if len(stack) < 8:
                    continue
                clip = clip.unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(clip)[1]
                    
                    # calculate the FLOPs
                    # =====================================
                    if mearue_flops:
                        flops_key_tsn = f"human-activity-classification-rgb-{(1, 3, 8, 224, 224)}"
                        if flops_key_tsn in flops_cache:
                            flops_tsn = flops_cache[flops_key_tsn]
                        else:
                            print()
                            calcutae_device = torch.device("cpu")
                            
                            # TSN
                            tsn_model = model
                            tsn_model.eval()
                            input_data = torch.randn(1, 3, 8, 224, 224)
                            tsn_model = tsn_model.to(calcutae_device)
                            input_data = input_data.to(calcutae_device)
                            flops_tsn, _ = profile(tsn_model, inputs=(input_data, ), verbose=False)
                            flops_tsn = flops_tsn / 1e9
                            
                            # save the FLOPs to the cache
                            flops_cache[flops_key_tsn] = flops_tsn
                            save_flops_cache_with_lock(flops_cache_path, flops_cache, flops_key_tsn)
                            
                            tsn_model = tsn_model.to(device)
                            
                        gflops += flops_tsn
                            
                    # =====================================
                
                _, preds = torch.max(outputs, 1)
                
                # record the label of each set of frames
                label_of_each_set_of_frames.append(
                    label_map["a"+f"{preds.item()+1:02d}"]
                )
                
            if cap.end:
                break
            
        # get the most common label of the set of frames
        label_of_this_video = Counter(label_of_each_set_of_frames).most_common(1)[0][0]
        # append the label of this video to the result
        result_label_of_this_video.append(label_of_this_video)
        result_label.append(label_of_this_video)
        
        pre_sign = pre_signs[video_path_idx]
        post_sign = generate_a_sign()
        
        result_pre_sign.append(pre_sign)
        result_post_sign.append(post_sign)
        
        # if use_cache is True, then save the cache
        if use_cache:
            cache["human-activity-classification-rgb"][video_path] = {
                "label": result_label_of_this_video
            }
            save_cache_with_lock(execution_cache_path, cache, "human-activity-classification-rgb", video_path)
        
    end_time_inference = time.perf_counter()
    
    print_tips("Human-activity-classification-rgb Task Ends.", emoji="tada", text_color="yellow", border=False)
    
    # record the time of each stage of the tool execution
    time_of_each_stage_of_tool_execution["pre_processing_time"] = end_time_preprocessing - start_time_preprocessing
    time_of_each_stage_of_tool_execution["model_loading_time"] = end_time_model_loading - start_time_model_loading
    time_of_each_stage_of_tool_execution["model_inference_time"] = end_time_inference - start_time_inference
    
    # record the FLOPs of the tool
    if mearue_flops:
        gflops_of_tool["gflops"] = gflops
    
    # construct the results
    results = {
        "tool_name": "human-activity-classification-rgb",
        "output": {
            "label": result_label,
            "pre_sign": result_pre_sign,
            "post_sign": result_post_sign
        },
        "saving": True,
        "execute_next": True
    }
    
    # if there is no output then set execute_next to False
    if len(result_label) == 0:
        results["execute_next"] = False
    
    return results

# (2) Depth modality
# model tool: object-detection-depth
def model_tool_object_detection_depth(
    args,
    time_of_each_stage_of_tool_execution,
    device=device,
    specified_saving_path=None, 
    gflops_of_tool=None
):
    '''
    This function is the implementation of the object-detection-depth tool.

    Args:
        args (dict): The arguments of the tool.
        time_of_each_stage_of_tool_execution (dict): The time of each stage of the tool execution.
        device (torch.device, optional): The device to use. Defaults to device.
        specified_saving_path (str, optional): It specifies the path to save the output of the tool. Defaults to None.

    Returns:
        dict: The result of the tool execution.
    '''
    if not args["execute_next"]:
        print_tips("Object-detection-depth skips.", emoji="zap", text_color="yellow", border=False)
        return {
            "task": "object-detection-depth",
            "output": {
                "label": [],
                "pre_sign": [],
                "post_sign": [],
            },
            "execute_next": False
        }
    
    print_tips("Object-detection-depth Task Starts.", emoji="zap", text_color="yellow", border=False)
    
    saving_folder = config["execution_settings"]["tool_name_saving_folder_map"][dataset_name]["object-detection-depth"]
    
    # check if the specified_saving_path is given
    # if it is not given, then save the output in the default path
    # otherwise, save the output in the specified path
    if specified_saving_path is None:
        saving_path = os.path.join(intermediate_output_path, saving_folder)
    else:
        saving_path = specified_saving_path
        
    os.makedirs(intermediate_output_path, exist_ok=True)
    os.makedirs(saving_path, exist_ok=True)
        
    # pre-process the input
    start_time_preprocessing = time.time()
    video_paths = args["output"]["depth_video"] # the list of the paths of the input videos
    pre_signs = args["output"]["post_sign"] # the signs of the previous tool calls' outputs
    end_time_preprocessing = time.time()
    
    # load the model
    start_time_model_loading = time.time()
    model = YOLO(config["execution_settings"]["model_checkpoint_paths"][dataset_name]["object_detection_depth"])
    tracker = DeepSort(max_age=30, embedder="mobilenet")
    end_time_model_loading = time.time()
    
    # inference
    start_time_inference = time.perf_counter()
    
    result_video_paths = [] # list to store the output cropped videos' paths
    result_label = [] # list to store the output labels
    result_pre_sign = [] # list to store the signs of the previous tool calls' outputs
    result_post_sign = [] # list to store the signs of the output of this tool call
    batch_size = 64 # batch size for the inference
    
    # if use_cache is True, then load the cache
    if use_cache:
        cache = load_cache_with_lock(execution_cache_path)
        
    if mearue_flops:
        gflops = 0.0
        flops_cache = load_cache_with_lock(flops_cache_path)
    
    for video_path_idx in range(len(video_paths)):
        video_path = video_paths[video_path_idx]
        
        # if use_cache is True and the video path is in the cache
        # then skip the video and directly use the results in the cache
        if use_cache:
            if video_path in cache["object-detection-depth"]:
                print_tips(f"[object-detection-depth] Video {video_path} has been processed before. Skip.", emoji="zap", text_color="yellow", border=False)
                result_video_paths.extend(cache["object-detection-depth"][video_path]["depth_video"])
                result_label.extend(cache["object-detection-depth"][video_path]["label"])
                
                # create the pre_sign and post_sign for the skipped video
                # and append them to the result dictionary
                for _ in cache["object-detection-depth"][video_path]["depth_video"]:
                    result_pre_sign.append(pre_signs[video_path_idx])
                    result_post_sign.append(generate_a_sign())
                    
                continue
        
        result_video_paths_of_this_video = [] # list to store the output cropped videos' paths of this video
        result_label_of_this_video = [] # list to store the output labels of this video
        map_from_track_id_to_cv2_writer = {} # map from track id to cv2 writer
        map_from_track_id_to_cropped_video_path = {} # map from track id to cropped video path
        
        # extract the time information from the video path
        date = video_path.split("/")[-2]
        time_point = video_path.split("/")[-1].split(".")[0]
        
        # read the video and get the information of the video
        frame_width, frame_height, fps, frame_count = get_video_info(video_path)
        
        print_tips(f"[object-detection-depth] Video Information: Frame Width: {frame_width}, Frame Height: {frame_height}, FPS: {fps}, Frame Count: {frame_count}, Path: {video_path}", emoji="zap", text_color="yellow", border=False)
        
        if any([frame_width == 0, frame_height == 0, fps == 0, frame_count == 0]):
            continue
        
        # clear all the tracks
        tracker.delete_all_tracks()
        
        # start to process the video
        cap = ObjectDetectionVideoCapture(video_path)
        processed_frame_count = 0
        while not cap.end:
            _, _ = cap.read()
            # if the stack is full or the video ends
            if (len(cap.stack) == batch_size) or (cap.end and len(cap.stack) != 0):
                _, frame_list = cap.get_video_clip()
                processed_frame_count += len(frame_list)
                
                sys.stdout.write(f"\r[object-detection-depth] Processed Frames: {processed_frame_count}/{frame_count} = {processed_frame_count/frame_count:.2f}")
                sys.stdout.flush()
                
                with torch.no_grad():
                    detection_list = model(frame_list, verbose=False)
                    
                    # =====================================
                    # calculate the FLOPs
                    if mearue_flops:
                        calcutae_device = torch.device("cpu")
                        
                        # YOLO
                        flops_key_yolo = f"object-detection-depth-yolo-{(len(frame_list), 3, 256, 256)}"
                        if flops_key_yolo in flops_cache:
                            flops_yolo = flops_cache[flops_key_yolo]
                            
                            print(flops_key_yolo, " Hit.")
                        else:
                            yolo_model = model.model
                            yolo_model.eval()
                            input_data = torch.randn(len(frame_list), 3, 256, 256)
                            yolo_model = yolo_model.to(calcutae_device)
                            input_data = input_data.to(calcutae_device)
                            flops_yolo, _ = profile(yolo_model, inputs=(input_data, ), verbose=False)
                            flops_yolo = flops_yolo / 1e9
                            
                            # save the FLOPs to the cache
                            flops_cache[flops_key_yolo] = flops_yolo
                            save_flops_cache_with_lock(flops_cache_path, flops_cache, flops_key_yolo)
                            
                            yolo_model = yolo_model.to(device)
                        
                        gflops += flops_yolo
                        
                        # DeepSort
                        flops_key_deepsort = f"object-detection-depth-tracker-{(len(frame_list), 3, 224, 224)}"
                        if flops_key_deepsort in flops_cache:
                            flops_tracker = flops_cache[flops_key_deepsort]
                            
                            print(flops_key_deepsort, " Hit.")
                        else:
                            tracker_model = tracker.embedder.model
                            tracker_model.eval()
                            # input should be torch.HalfTensor
                            input_data_embedder = torch.randn(len(frame_list), 3, 224, 224).half()
                            tracker_model = tracker_model.to(calcutae_device)
                            input_data_embedder = input_data_embedder.to(calcutae_device)
                            flops_tracker, _ = profile(tracker_model, inputs=(input_data_embedder, ), verbose=False)
                            flops_tracker = flops_tracker / 1e9
                            
                            # save the FLOPs to the cache
                            flops_cache[flops_key_deepsort] = flops_tracker
                            save_flops_cache_with_lock(flops_cache_path, flops_cache, flops_key_deepsort)
                            
                            tracker_model = tracker_model.to(device)
                            
                        gflops += flops_tracker
                    # =====================================
                    
                    
                formatted_detections = []
                class_names = detection_list[0].names
                
                for detection in detection_list:
                    boxes = detection.boxes
                    formatted_detections.append({
                        "boxes": boxes.xyxy,
                        "scores": boxes.conf,
                        "labels": boxes.cls,
                    })
                    
                # convert the detection to the format required by the tracker
                # in this case, we only focus on the person class
                focused_classes = [0]
                convert_detection_partial = partial(
                    convert_detection_object_detection, 
                    threshold=0.70, 
                    classes=focused_classes)
                with ThreadPoolExecutor() as executor:
                    detection_list = list(executor.map(
                        convert_detection_partial, 
                        formatted_detections))
                    
                for i in range(len(detection_list)):
                    detection = detection_list[i]
                    frame = frame_list[i]
                    tracks = tracker.update_tracks(detection, frame=frame)
                    
                    for track in tracks:
                        x1, y1, x2, y2 = track.to_tlbr()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        track_id = track.track_id
                        
                        # if a new track is found
                        if track_id not in map_from_track_id_to_cv2_writer:
                            saving_name = f"{date}_{time_point}_object_{track_id}_{class_names[int(track.det_class)]}.avi"
                            obj_video_path = os.path.join(
                                saving_path,
                                saving_name
                            )
                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            
                            # create the cv2 writer
                            # and put it in the map
                            map_from_track_id_to_cv2_writer[track_id] = cv2.VideoWriter(
                                obj_video_path,
                                fourcc,
                                fps,
                                (640, 480)
                            )
                            
                            # put the video path in the map
                            map_from_track_id_to_cropped_video_path[track_id] = obj_video_path
                            
                            # obtain the pre_sign and post_sign
                            pre_sign = pre_signs[video_path_idx]
                            post_sign = generate_a_sign()
                            
                            # append the new results
                            result_video_paths.append(obj_video_path)
                            result_label.append(class_names[int(track.det_class)])
                            result_pre_sign.append(pre_sign)
                            result_post_sign.append(post_sign)
                            result_video_paths_of_this_video.append(obj_video_path)
                            result_label_of_this_video.append(class_names[int(track.det_class)])
                            
                        # crop the object frame and set the output height and width
                        obj_video_frame = frame[y1:y2, x1:x2]
                        output_height, output_width = 480, 640
                        
                        # if the object frame is not empty
                        # then write the object frame to the cv2 writer
                        if obj_video_frame.size != 0:
                            height_of_obj_video_frame, width_of_obj_video_frame = obj_video_frame.shape[:2]
                            scale = min(output_height / height_of_obj_video_frame, output_width / width_of_obj_video_frame)
                            
                            scaled_width = int(width_of_obj_video_frame * scale)
                            scaled_height = int(height_of_obj_video_frame * scale)
                            scaled_obj_video_frame = cv2.resize(obj_video_frame, (scaled_width, scaled_height))
                            
                            output_scaled_obj_video_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                            
                            x_offset = (output_width - scaled_width) // 2
                            y_offset = (output_height - scaled_height) // 2
                            
                            output_scaled_obj_video_frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = scaled_obj_video_frame
                            map_from_track_id_to_cv2_writer[track_id].write(output_scaled_obj_video_frame)
        
        
        # if use_cache is True, then save the cache
        if use_cache:
            cache["object-detection-depth"][video_path] = {
                "depth_video": result_video_paths_of_this_video,
                "label": result_label_of_this_video
            }
            save_cache_with_lock(execution_cache_path, cache, "object-detection-depth", video_path)
                      
        cap.release()
        for writer in map_from_track_id_to_cv2_writer.values():
            writer.release()
            
    end_time_inference = time.perf_counter()
    
    print_tips("Object-detection-depth Task Ends.", emoji="tada", text_color="yellow", border=False)
    
    # record the time of each stage of the tool execution
    time_of_each_stage_of_tool_execution["pre_processing_time"] = end_time_preprocessing - start_time_preprocessing
    time_of_each_stage_of_tool_execution["model_loading_time"] = end_time_model_loading - start_time_model_loading
    time_of_each_stage_of_tool_execution["model_inference_time"] = end_time_inference - start_time_inference
    
    # record the FLOPs of the tool
    if mearue_flops:
        gflops_of_tool["gflops"] = gflops
    
    # construct the results
    results = {
        "tool_name": "object-detection-depth",
        "output": {
            "depth_video": result_video_paths,
            "label": result_label,
            "pre_sign": result_pre_sign,
            "post_sign": result_post_sign
        },
        "saving": True,
        "execute_next": True
    }
    
    # if there is no output then set execute_next to False
    if len(result_video_paths) == 0:
        results["execute_next"] = False
    
    return results

# model tool: human-activity-classification-depth
def model_tool_human_activity_classification_depth(
    args,
    time_of_each_stage_of_tool_execution,
    device=device,
    gflops_of_tool=None
):
    '''
    This function is the implementation of the human-activity-classification-depth tool.

    Args:
        args (dict): The arguments of the tool.
        time_of_each_stage_of_tool_execution (dict): The time of each stage of the tool execution.
        device (torch.device, optional): The device to use. Defaults to device.

    Returns:
        dict: The result of the tool execution.
    '''
    if not args["execute_next"]:
        print_tips("Human-activity-classification-depth skips.", emoji="zap", text_color="yellow", border=False)
        return {
            "task": "human-activity-classification-depth",
            "output": {
                "label": [],
                "pre_sign": [],
                "post_sign": [],
            },
            "execute_next": False
        }
    
    print_tips("Human-activity-classification-depth Task Starts.", emoji="zap", text_color="yellow", border=False)
    
    start_time_preprocessing = time.perf_counter()
    video_paths = args["output"]["depth_video"]
    pre_signs = args["output"]["post_sign"]
    end_time_preprocessing = time.perf_counter()
    
    start_time_model_loading = time.perf_counter()
    model = TSN(num_class=30, dropout=0.8, modality='RGB', is_shift=True).eval()
    model.load_state_dict(torch.load(config["execution_settings"]["model_checkpoint_paths"][dataset_name]["human_activity_classification_rgb"]))
    model = model.to(device)
    
    label_map = {
        "a01": "boxing",
        "a02": "push",
        "a03": "pat on shoulder",
        "a04": "handshake",
        "a05": "face-to-face conversation",
        "a06": "standing still",
        "a07": "waving arms",
        "a08": "playing with phone using both hands",
        "a09": "rubbing hands",
        "a10": "touching head",
        "a11": "coughing",
        "a12": "jumping with both feet",
        "a13": "stomping feet",
        "a14": "walking",
        "a15": "squatting down",
        "a16": "bending over",
        "a17": "throwing things",
        "a18": "picking up things",
        "a19": "kicking cabinet",
        "a20": "looking for something in the cabinet",
        "a21": "angrily slapping the cabinet",
        "a22": "sitting still",
        "a23": "sitting and making a phone call",
        "a24": "standing up",
        "a25": "sitting down",
        "a26": "raising hand to check time/putting down",
        "a27": "stretching",
        "a28": "taking off mask/putting on mask",
        "a29": "lying on a mat",
        "a30": "falling down/getting up"
    }
    end_time_model_loading = time.perf_counter()
    
    start_time_inference = time.perf_counter()
    result_label = []
    result_pre_sign = []
    result_post_sign = []
    
    # if use_cache is True, then load the cache
    if use_cache:
        cache = load_cache_with_lock(execution_cache_path)
        
    if mearue_flops:
        gflops = 0.0
        flops_cache = load_cache_with_lock(flops_cache_path)
    
    for video_path_idx in range(len(video_paths)):
        video_path = video_paths[video_path_idx]
        
        # if use_cache is True and the video path is in the cache
        # then skip the video and directly use the results in the cache
        if use_cache:
            if video_path in cache["human-activity-classification-depth"]:
                print_tips(f"[human-activity-classification-depth] Video {video_path} has been processed before. Skip.", emoji="zap", text_color="yellow", border=False)
                result_label.extend(cache["human-activity-classification-depth"][video_path]["label"])
                
                # create the pre_sign and post_sign for the skipped video
                # and append them to the result dictionary
                for _ in cache["human-activity-classification-depth"][video_path]["label"]:
                    result_pre_sign.append(pre_signs[video_path_idx])
                    result_post_sign.append(generate_a_sign())
                
                continue
        
        label_of_each_set_of_frames = []
        label_of_this_video = []
        result_label_of_this_video = []
        
        # read the video and get the information of the video
        frame_width, frame_height, fps, frame_count = get_video_info(video_path)
        
        # if the total frame count is less than 8, then skip the video
        if frame_count < 8:
            continue
        
        print_tips(f"[human-activity-classification-depth] Video Information: Frame Width: {frame_width}, Frame Height: {frame_height}, FPS: {fps}, Frame Count: {frame_count}, Path: {video_path}", emoji="zap", text_color="yellow", border=False)
        
        cap = HARVideoCapture(video_path)
        batch_size = 32
        
        while not cap.end:
            _, _ = cap.read()
            # if the stack is full or the video ends
            if len(cap.stack) == batch_size or (cap.end and len(cap.stack) != 0):
                clip, stack = cap.get_video_clip()
                if len(stack) < 8:
                    continue
                clip = clip.unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(clip)[1]
                    
                    # calculate the FLOPs
                    # =====================================
                    if mearue_flops:
                        flops_key_tsn = f"human-activity-classification-depth-{(1, 3, 8, 224, 224)}"
                        if flops_key_tsn in flops_cache:
                            flops_tsn = flops_cache[flops_key_tsn]
                        else:
                            print()
                            calcutae_device = torch.device("cpu")
                            
                            # TSN
                            tsn_model = model
                            tsn_model.eval()
                            input_data = torch.randn(1, 3, 8, 224, 224)
                            tsn_model = tsn_model.to(calcutae_device)
                            input_data = input_data.to(calcutae_device)
                            flops_tsn, _ = profile(tsn_model, inputs=(input_data, ), verbose=False)
                            
                            flops_tsn = flops_tsn / 1e9
                            
                            # save the FLOPs to the cache
                            flops_cache[flops_key_tsn] = flops_tsn
                            save_flops_cache_with_lock(flops_cache_path, flops_cache, flops_key_tsn)
                            
                            tsn_model = tsn_model.to(device)
                            
                        gflops += flops_tsn
                    # =====================================
                
                _, preds = torch.max(outputs, 1)
                
                # record the label of each set of frames
                label_of_each_set_of_frames.append(
                    label_map["a"+f"{preds.item()+1:02d}"]
                )
            
            if cap.end:
                break
        
        # get the most common label of the set of frames
        label_of_this_video = Counter(label_of_each_set_of_frames).most_common(1)[0][0]
        # append the label of this video to the result
        result_label_of_this_video.append(label_of_this_video)
        result_label.append(label_of_this_video)
        
        pre_sign = pre_signs[video_path_idx]
        post_sign = generate_a_sign()
        
        result_pre_sign.append(pre_sign)
        result_post_sign.append(post_sign)
        
        # if use_cache is True, then save the cache
        if use_cache:
            cache["human-activity-classification-depth"][video_path] = {
                "label": result_label_of_this_video
            }
            save_cache_with_lock(execution_cache_path, cache, "human-activity-classification-depth", video_path)
        
    end_time_inference = time.perf_counter()
    
    print_tips("Human-activity-classification-depth Task Ends.", emoji="tada", text_color="yellow", border=False)
    
    # record the time of each stage of the tool execution
    time_of_each_stage_of_tool_execution["pre_processing_time"] = end_time_preprocessing - start_time_preprocessing
    time_of_each_stage_of_tool_execution["model_loading_time"] = end_time_model_loading - start_time_model_loading
    time_of_each_stage_of_tool_execution["model_inference_time"] = end_time_inference - start_time_inference
    
    # record the FLOPs of the tool
    if mearue_flops:
        gflops_of_tool["gflops"] = gflops
    
    # construct the results
    results = {
        "tool_name": "human-activity-classification-depth",
        "output": {
            "label": result_label,
            "pre_sign": result_pre_sign,
            "post_sign": result_post_sign
        },
        "saving": True,
        "execute_next": True
    }
    
    # if there is no output then set execute_next to False
    if len(result_label) == 0:
        results["execute_next"] = False
    
    return results

# 2. database tools
# database tool: query-video-data-rgb
def database_tool_query_video_data_rgb(
    args,
    time_of_each_stage_of_tool_execution,
    gflops_of_tool=None
):
    '''
    This function is the implementation of the query-video-data-rgb tool.
    Query the video data in the database according to the given time range.

    Args:
        args (dict): The arguments of the tool.
        time_of_each_stage_of_tool_execution (dict): The time of each stage of the tool execution.
        
    Returns:
        dict: The result of the tool execution.
    '''
    
    if not args["execute_next"]:
        print_tips("Query-video-data-rgb skips.", emoji="zap", text_color="yellow", border=False)
        return {
            "task": "query-video-data-rgb",
            "output": {
                "rgb_video": [],
                "pre_sign": [],
                "post_sign": [],
            },
            "execute_next": False
        }
        
    print_tips("Query-video-data-rgb Task Starts.", emoji="zap", text_color="yellow", border=False)
    
    databse_path = config["execution_settings"]["sensor_database_path"][dataset_name]["rgb"]
    # query_condition list contains multiple query conditions
    # each query condition is a list of the form [date, start_time, end_time]
    # e.g. query_condition_list = [[20240101, 0, 6]]
    query_condition_list = args["output"]["text"]
    
    # check if the query_condition_list is a list
    # if it is not a list, then convert it to a list
    if not isinstance(query_condition_list, list):
        query_condition_list = [query_condition_list]
    
    pre_signs = args["output"]["post_sign"]
    
    result_video_paths = []
    result_pre_sign = []
    result_post_sign = []
    
    try:
        for query_condition_idx in range(len(query_condition_list)):
            query_condition = query_condition_list[query_condition_idx]
            date = query_condition[0]
            start_time = query_condition[1]
            end_time = query_condition[2]
            
            day_path = databse_path + str(date).zfill(8) + "/"
            
            for file_name in os.listdir(day_path):
                time_point = int(file_name.split(".")[0][0:2])
                
                if start_time <= time_point <= end_time:
                    video_path = day_path + file_name
                    pre_sign = pre_signs[query_condition_idx]
                    post_sign = generate_a_sign()
                    
                    result_video_paths.append(video_path)
                    result_pre_sign.append(pre_sign)
                    result_post_sign.append(post_sign)
            
    except Exception as e:
        print_tips(str(e), emoji="warning", text_color="red", border=False)
        raise "Error occurs in the query-video-data-rgb task."
    
    print_tips("Query-video-data-rgb Task Ends.", emoji="tada", text_color="yellow", border=False)
    
    # construct the results
    results = {
        "tool_name": "query-video-data-rgb",
        "output": {
            "rgb_video": result_video_paths,
            "pre_sign": result_pre_sign,
            "post_sign": result_post_sign
        },
        "saving": True,
        "execute_next": True,
    }
    
    if len(results["output"]["rgb_video"]) == 0:
        results["execute_next"] = False
    
    return results

# database tool: query-video-data-depth
def database_tool_query_video_data_depth(
    args,
    time_of_each_stage_of_tool_execution,
    gflops_of_tool=None
):
    '''
    This function is the implementation of the query-video-data-depth tool.
    Query the video data in the database according to the given time range.

    Args:
        args (dict): The arguments of the tool.
        time_of_each_stage_of_tool_execution (dict): The time of each stage of the tool execution.
        
    Returns:
        dict: The result of the tool execution.
    '''
    
    if not args["execute_next"]:
        print_tips("Query-video-data-depth skips.", emoji="zap", text_color="yellow", border=False)
        return {
            "task": "query-video-data-depth",
            "output": {
                "depth_video": [],
                "pre_sign": [],
                "post_sign": [],
            },
            "execute_next": False
        }
        
    print_tips("Query-video-data-depth Task Starts.", emoji="zap", text_color="yellow", border=False)
    
    database_path = config["execution_settings"]["sensor_database_path"][dataset_name]["dep"]
    # query_condition list contains multiple query conditions
    # each query condition is a list of the form [date, start_time, end_time]
    # e.g. query_condition_list = [[20240101, 0, 6]]
    query_condition_list = args["output"]["text"]
    
    # check if the query_condition_list is a list
    # if it is not a list, then convert it to a list
    if not isinstance(query_condition_list, list):
        query_condition_list = [query_condition_list]
        
    pre_signs = args["output"]["post_sign"]
    
    result_video_paths = []
    result_pre_sign = []
    result_post_sign = []
    
    try:
        for query_condition_idx in range(len(query_condition_list)):
            query_condition = query_condition_list[query_condition_idx]
            date = query_condition[0]
            start_time = query_condition[1]
            end_time = query_condition[2]
            
            day_path = database_path + str(date).zfill(8) + "/"
            
            for file_name in os.listdir(day_path):
                time_point = int(file_name.split(".")[0][0:2])
                
                if start_time <= time_point <= end_time:
                    video_path = day_path + file_name
                    pre_sign = pre_signs[query_condition_idx]
                    post_sign = generate_a_sign()
                    
                    result_video_paths.append(video_path)
                    result_pre_sign.append(pre_sign)
                    result_post_sign.append(post_sign)
                    
    except Exception as e:
        print_tips(str(e), emoji="warning", text_color="red", border=False)
        raise "Error occurs in the query-video-data-depth task."
    
    print_tips("Query-video-data-depth Task Ends.", emoji="tada", text_color="yellow", border=False)
    
    # construct the results
    results = {
        "tool_name": "query-video-data-depth",
        "output": {
            "depth_video": result_video_paths,
            "pre_sign": result_pre_sign,
            "post_sign": result_post_sign
        },
        "saving": True,
        "execute_next": True,
    }
    
    if len(results["output"]["depth_video"]) == 0:
        results["execute_next"] = False
        
    return results

# 3. sensor tools 
def sensor_tool_tplink_start_recording(
    args
):
    '''
    This function is the implementation of the start-recording tool.

    Args:
        args (dict): The arguments of the tool. It should be a list of the form [sensor_id]. sensor_id refers to the id of the sensor. (e.g. [1]).
        
    Returns:
        dict: The result of the tool execution. It contains one label list. It contains the status of the tool execution. The status collection is ['success', 'fail'].
    '''
    
    if not args["execute_next"]:
        print_tips("Tplink-start-recording skips.", emoji="zap", text_color="yellow", border=False)
        return {
            "tool_name": "tplink-start-recording",
            "output": {
                "pre_sign": [],
                "post_sign": [],
                "label": []
            },
            "execute_next": False
        }
        
    print_tips("Tplink-start-recording Task Starts.", emoji="zap", text_color="yellow", border=False)
    
    sensor_id = args["output"]["text"][0]
    
    ip_of_sensor = config["sensor_settings"][dataset_name][sensor_id]["ip"]
    port = config["sensor_settings"][dataset_name][sensor_id]["port"]
    username = config["sensor_settings"][dataset_name][sensor_id]["username"]
    password = config["sensor_settings"][dataset_name][sensor_id]["password"]
    data = str('{"method":"set","record_plan":{"chn1_channel":{"enabled":"on","monday":"%5b%220000-2400%3a1%22%5d","tuesday":"%5b%220000-2400%3a1%22%5d","wednesday":"%5b%220000-2400%3a1%22%5d","thursday":"%5b%220000-2400%3a1%22%5d","friday":"%5b%220000-2400%3a1%22%5d","saturday":"%5b%220000-2400%3a1%22%5d","sunday":"%5b%220000-2400%3a1%22%5d"}}}')
    
    # tunnel to the sensor
    B_ip = '#'  # ip of the tunnel server
    B_username = '#'  # username of the tunnel server
    B_password = '#'  # password of the tunnel server
    
    local_bind_port = 8080  # local port to bind
    
    with SSHTunnelForwarder(
        (B_ip, 22),
        ssh_username=B_username,
        ssh_password=B_password,
        remote_bind_address=(ip_of_sensor, port),
        local_bind_address=('0.0.0.0', local_bind_port)
    ) as tunnel:
        base_url = f"http://127.0.0.1:{local_bind_port}"

        # 
        response = post_data(
            base_url,
            data,
            get_stok(
                base_url,
                username,
                password
            )
        )
        
    print_tips("Tplink-start-recording Task Ends.", emoji="tada", text_color="yellow", border=False)
    
    # construct the results
    results = {
        "tool_name": "tplink-start-recording",
        "output": {
            "pre_sign": args["output"]["post_sign"],
            "post_sign": [generate_a_sign()],
            "label": [
                "success" if response["error_code"] == 0 else "fail"
            ]
        },
        "saving": False,
        "execute_next": True,
    }
    
    return results


def sensor_tool_tplink_start_manual_alarm(
    args
):
    '''
    This function is the implementation of the start-manual-alarm tool.

    Args:
        args (dict): The arguments of the tool. It should be a list of the form [sensor_id]. sensor_id refers to the id of the sensor. (e.g. [1]).
        
    Returns:
        dict: The result of the tool execution. It contains one label list. It contains the status of the tool execution. The status collection is ['success', 'fail'].
    '''
    
    if not args["execute_next"]:
        print_tips("Tplink-start-manual-alarm skips.", emoji="zap", text_color="yellow", border=False)
        return {
            "tool_name": "tplink-start-manual-alarm",
            "output": {
                "pre_sign": [],
                "post_sign": [],
                "label": []
            },
            "execute_next": False
        }

    print_tips("Tplink-start-manual-alarm Task Starts.", emoji="zap", text_color="yellow", border=False)

    sensor_id = args["output"]["text"][0]

    ip_of_sensor = config["sensor_settings"][dataset_name][sensor_id]["ip"]
    port = config["sensor_settings"][dataset_name][sensor_id]["port"]
    username = config["sensor_settings"][dataset_name][sensor_id]["username"]
    password = config["sensor_settings"][dataset_name][sensor_id]["password"]
    data = str('{"method":"do","msg_alarm":{"manual_msg_alarm":{"action":"start"}}}')
    
    # tunnel to the sensor
    B_ip = '#'  # ip of the tunnel server
    B_username = '#'  # username of the tunnel server
    B_password = '#'  # password of the tunnel server

    local_bind_port = 8080  # local port to bind
    
    with SSHTunnelForwarder(
        (B_ip, 22),
        ssh_username=B_username,
        ssh_password=B_password,
        remote_bind_address=(ip_of_sensor, port),
        local_bind_address=('0.0.0.0', local_bind_port)
    ) as tunnel:
        base_url = f"http://127.0.0.1:{local_bind_port}"

        # 
        response = post_data(
            base_url,
            data,
            get_stok(
                base_url,
                username,
                password
            )
        )

    print_tips("Tplink-start-manual-alarm Task Ends.", emoji="tada", text_color="yellow", border=False)
    
    # construct the results
    results = {
        "tool_name": "tplink-start-manual-alarm",
        "output": {
            "pre_sign": args["output"]["post_sign"],
            "post_sign": [generate_a_sign()],
            "label": [
                "success" if response["error_code"] == 0 else "fail"
            ]
        },
        "saving": False,
        "execute_next": True,
    }
    
    return results


def sensor_tool_microphone_start_recoding(
    args
):
    '''
    This function is the implementation of the start-recording tool.

    Args:
        args (dict): The arguments of the tool. It should be a list of the form [sensor_id]. sensor_id refers to the id of the sensor. (e.g. [1]).
        
    Returns:
        dict: The result of the tool execution. It contains one label list. It contains the status of the tool execution. The status collection is ['success', 'fail'].
    '''
    
    if not args["execute_next"]:
        print_tips("Microphone-start-recording skips.", emoji="zap", text_color="yellow", border=False)
        
        return {
            "tool_name": "microphone-start-recording",
            "output": {
                "pre_sign": [],
                "post_sign": [],
                "label": []
            },
            "execute_next": False
        }
        
    print_tips("Microphone-start-recording Task Starts.", emoji="zap", text_color="yellow", border=False)
    
    server_ip = str(config["sensor_settings"][dataset_name]["microphone_room_1"]["ip"])
    username = str(config["sensor_settings"][dataset_name]["microphone_room_1"]["username"])
    password = str(config["sensor_settings"][dataset_name]["microphone_room_1"]["password"])
    
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=server_ip, username=username, password=password)
        stdin, stdout, stderr = client.exec_command('bash '+ config["sensor_settings"][dataset_name]["microphone_room_1"]["start_recording_script_path"])
        output = stdout.read().decode()
        error = stderr.read().decode()
        
        client.close()
        
        
        print_tips("Microphone-start-recording Task Ends.", emoji="tada", text_color="yellow", border=False)
        return output, error

    except Exception as e:
        print_tips(str(e), emoji="warning", text_color="red", border=False)
        return None, str(e)

# define a map of tool names to their implementations
tool_name_tool_implementation_map = {
    # RGB modality
    "query-video-data-rgb": database_tool_query_video_data_rgb,
    "object-detection-rgb": model_tool_object_detection_rgb,
    "human-activity-classification-rgb": model_tool_human_activity_classification_rgb,
    
    # Depth modality
    "query-video-data-depth": database_tool_query_video_data_depth,
    "object-detection-depth": model_tool_object_detection_depth,
    "human-activity-classification-depth": model_tool_human_activity_classification_depth,
    
    # Sensor Control
    "take-picture": sensor_tool_tplink_start_recording,
    "send-notification": sensor_tool_tplink_start_manual_alarm,
    "record-microphone": sensor_tool_microphone_start_recoding
}

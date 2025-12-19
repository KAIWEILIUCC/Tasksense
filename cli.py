import os

# set the environment variable "CONFIG_PATH" to the path of the configuration file
os.environ['CONFIG_PATH'] = "./config/config.yaml"

import torch

from src.coordinator import coordinating
from src.process_pool import DataStructurePool
from src.utils import (print_tips, visualize_agent_response)



def cli():
    print_tips("Welcome!", text_color="green", emoji="smiley")
    
    # init_execution_cache()
    # init_assessment_cache()
    # empty_output_folder()
    
    # set the start method of the multiprocessing module to "spawn"
    torch.multiprocessing.set_start_method("spawn")
    
    # create a data structure pool
    data_structure_pool = DataStructurePool()
    # warm up the pool
    data_structure_pool.warmup_pool()
    
    while True:
        messages = []
        # get input from the user
        message = input("[ User ]: ")
        if message == "exit":
            break
        
        # add the input message to the list of messages
        messages.append({"role": "user", "content": message})
        
        # send messages to the coordinator
        final_response, results_of_each_stage = coordinating(
            messages=messages,
            data_structure_pool=data_structure_pool,
            return_planning_results=False,
            return_execution_results=False,
            skip_planning=False,
        )
        
        visualize_agent_response(final_response)
    
    print("CLI stopped.")
    
    # release the data structure pool
    data_structure_pool.release_pool()
    # destroy the data structure pool
    del data_structure_pool

if __name__ == "__main__":
    cli()

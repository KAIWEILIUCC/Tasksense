import time
from multiprocessing import Manager, Pool

from src.utils import print_tips

"""
This module defines a class DataStructurePool for managing a pool of data structures for storing the process pool and shared data structures between processes.
"""

# DataStructurePool: a pool of data structures for storing the process pool and shared data structures between processes
# An instance of this class can be shared by multiple processes
class DataStructurePool:
    # initialize the data structure pool
    def __init__(self):
        self.data_structure = self._create_data_structure()
    
    # create the data structure
    def _create_data_structure(self):
        # create a manager object
        manager = Manager()
        # create a list for storing tool calls
        tool_calls = manager.list()
        # create a dictionary for storing shared data
        d = manager.dict()
        # create a process pool
        # the number of processes in the pool is set to 3
        # too many processes may cause the system to run out of memory
        pool = Pool(processes=3)
        # create a dictionary for storing the execution time of each tool call
        execution_time_of_each_tool_call = manager.dict()
        # create a dictionary fir storing the GFLOPs of each tool call
        gflops_of_each_tool_call = manager.dict()
        
        # print("Data structure created.")
        print_tips("Data structure created.", text_color="yellow", emoji="white_check_mark", border=False)
        
        return tool_calls, d, pool, execution_time_of_each_tool_call, gflops_of_each_tool_call
    
    # acquire the data structure
    def acquire(self):
        return self.data_structure
    
    # clear the data structure,
    # i.e., clear the list for storing tool calls, clear the dictionary for storing shared data, and clear the dictionary for storing the execution time of each tool call
    def clear(self):
        tool_calls, d, pool, execution_time_of_each_tool_call, gflops_of_each_tool_call = self.data_structure
        tool_calls[:] = []
        d.clear()
        execution_time_of_each_tool_call.clear()
        gflops_of_each_tool_call.clear()
    
    # release the data structure
    def release_pool(self):
        _, _, pool, _, _ = self.data_structure
        pool.close()
        pool.join()
    
    # warm up the pool
    # this function is used to warm up the pool by executing some dummy tool calls
    # by doing this, we can avoid the overhead of creating new processes when executing the first tool call
    def warmup_pool(self, num_dummy_tool_calls=16):
        _, _, pool, _, _ = self.data_structure
        
        def dummy_tool_call():
            time.sleep(0.1)  # simulate the execution of a tool call
        
        warmup_results = []
        for _ in range(num_dummy_tool_calls):
            warmup_results.append(pool.apply_async(dummy_tool_call))
        
        for warmup_result in warmup_results:
            warmup_result.wait()
            
        print_tips("Process pool warmed up.", text_color="yellow", emoji="white_check_mark", border=False)
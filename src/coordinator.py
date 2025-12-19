import json
import time

from src.execution.executor import execute
from src.planning.plan_validator import check_plan
from src.planning.planner import planning
from src.responding.respondent import generate_response
from src.utils import print_tips, visualize_plan
from src.planning.checker import command_solvability_check


def coordinating(
    messages, # list of messages
    data_structure_pool, # data structure pool
    return_planning_results=False, # whether to return the planning results in advance
    return_execution_results=False, # whether to return the execution results in advance
    skip_planning=False, # whether to skip the planning stage
    specified_plan=None, # this parameter is used to specify a plan
):
    check_flag, check_response = command_solvability_check(
        user_input=messages[-1]["content"],
    )
    
    if check_flag == 'no':
        return "Command Not Solvable!", {}
    
    # create a dictionary to store the results of each stage
    results_of_each_stage = {
        "user_input": None,
        "planning_results": None,
        "execution_results": None,
        "final_response": None,
    }
    
    
    user_input = messages[-1]["content"]
    results_of_each_stage["user_input"] = user_input
    
    print_tips("Mission Starts", text_color="green", emoji="rocket")
    
    # **************************************************************************************************************************************************************
    # *                                                                                                                                                            *
    # *                                                                   Planning Stage Starts                                                                    *
    # *                                                                                                                                                            *
    # **************************************************************************************************************************************************************
    
    # if not skipping the planning stage
    if not skip_planning:
        print_tips("Planning Stage Starts", text_color="yellow", emoji="bulb")
        
        # get the generated plan, first generation
        planning_str = planning(user_input)
        
        # check the grammar of the plan, first check
        grammar_checking_result = check_plan(planning_str)
        
        # if the grammar checking fails
        if not grammar_checking_result['pass']:
            print_tips("Grammar Checking Failed: "+grammar_checking_result["error"], text_color="yellow", emoji="x")
            time.sleep(30) # marker: remove later
            # get the generated plan, second generation
            planning_str = planning(user_input)
            
            # check the grammar of the plan, second check
            grammar_checking_result = check_plan(planning_str)

            # if the grammar checking fails again
            if not grammar_checking_result["pass"]:
                print_tips("Grammar Checking Failed Again: "+grammar_checking_result["error"], text_color="yellow", emoji="x")
                time.sleep(30) # marker: remove later
                # get the generated plan, third generation
                planning_str = planning(user_input)
                # check the grammar of the plan, third check
                grammar_checking_result = check_plan(planning_str)
                
                # if the grammar checking fails again
                if not grammar_checking_result["pass"]:
                    print_tips("Grammar Checking Failed Again: "+grammar_checking_result["error"], text_color="yellow", emoji="x")
                    return "Grammar Checking Failed", {}
                
                # if the grammar checking passes after the third generation
                else:
                    print_tips("Grammar Checking Passed", text_color="yellow", emoji="white_check_mark", border=False)
                    plan = json.loads(planning_str)
                    
            # if the grammar checking passes after the second generation
            else:
                print_tips("Grammar Checking Passed", text_color="yellow", emoji="white_check_mark", border=False)
                plan = json.loads(planning_str)
                
        # if the grammar checking passes after the first generation
        else:
            print_tips("Grammar Checking Passed", text_color="yellow", emoji="white_check_mark", border=False)
            plan = json.loads(planning_str)
        
        # if the length of the plan is 0
        if len(plan) == 0:
            return "Empty Plan", {}
        
        # record the planning results
        results_of_each_stage["planning_results"] = plan
        
        # if returning the planning results in advance
        if return_planning_results:
            return plan, results_of_each_stage
        
        print("Generated Plan: ", plan)
        
        visualize_plan(plan)
        
        print_tips("Planning Stage Ends", text_color="yellow", emoji="bulb")
            
    # if skipping the planning stage
    else:
        print_tips("Skipping Planning Stage", text_color="yellow", emoji="fast_forward")
        
        if specified_plan is None:
            plan = [
                
            ]
        else:
            plan = specified_plan
            
        visualize_plan(plan)
        
        # record the planning results
        results_of_each_stage["planning_results"] = plan
        
        print_tips("Planning Stage Skipped", text_color="yellow", emoji="bulb")
    
    # **************************************************************************************************************************************************************
    # *                                                                                                                                                            *
    # *                                                                    Planning Stage Ends                                                                     *
    # *                                                                                                                                                            *
    # **************************************************************************************************************************************************************
    
    # **************************************************************************************************************************************************************
    # *                                                                                                                                                            *
    # *                                                                   Execution Stage Starts                                                                   *
    # *                                                                                                                                                            *
    # **************************************************************************************************************************************************************
    
    print_tips("Execution Stage Starts", text_color="yellow", emoji="hammer_and_wrench")
    
    execution_results, disabled_time_range = execute(
        user_input=user_input,
        data_structure_pool=data_structure_pool,
        plan=plan,
    )
    
    results_of_each_stage["execution_results"] = execution_results
    
    if return_execution_results:
        return execution_results, results_of_each_stage
    
    print_tips("Execution Stage Ends", text_color="yellow", emoji="hammer_and_wrench")
    # **************************************************************************************************************************************************************
    # *                                                                                                                                                            *
    # *                                                                    Execution Stage Ends                                                                    *
    # *                                                                                                                                                            *
    # **************************************************************************************************************************************************************
    
    # **************************************************************************************************************************************************************
    # *                                                                                                                                                            *
    # *                                                                    Response Stage Starts                                                                   *
    # *                                                                                                                                                            *
    # **************************************************************************************************************************************************************
    print_tips("Response Stage Starts", text_color="yellow", emoji="speech_balloon")
    
    
    response_results = generate_response(
        original_execution_results=execution_results,
        user_input=user_input,
        disabled_time_range=disabled_time_range,
    )
    
    results_of_each_stage["final_response"] = response_results
    
    print_tips("Response Stage Ends", text_color="yellow", emoji="speech_balloon")
    # **************************************************************************************************************************************************************
    # *                                                                                                                                                            *
    # *                                                                     Response Stage Ends                                                                    *
    # *                                                                                                                                                            *
    # **************************************************************************************************************************************************************
    
    print_tips("Mission Ends", text_color="green", emoji="party_popper")
    
    return response_results, results_of_each_stage
    
    

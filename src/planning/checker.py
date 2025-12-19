from src.api.llm_apis import send_request

prompt_map={
    'inlab': './src/check/prompt/prompt_inlab.txt',
}



def command_solvability_check(user_input=None,
                              dataset='inlab',
                              prompt_path='./src/check/prompt/prompt_inlab.txt',
                              activity_label_pth = './src/check/activity_label_list/inlab_activity.txt'):
    check_flag=None 
    
    with open(prompt_path, "r") as file:
        prompt_content = file.read()
    
    with open(activity_label_pth, "r") as file:
        activity_label_list = file.read()
    
    message=prompt_content.format(
        user_input=user_input,
        activity_label_list=activity_label_list
    )
    messages = []
    messages.append({"role": "user", "content": message})
    
    check_response = send_request(messages)
    if 'skip' in check_response: # **skip** or "this is an open question, skip"
        check_flag = 'skip'
    elif '**yes**' in check_response:
        check_flag = 'yes' 
    elif '**no**' in check_response: # be careful about this part
        check_flag = 'no'

    return check_flag, check_response
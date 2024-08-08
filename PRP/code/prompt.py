# -*- coding: utf-8 -*-
import time
import json
from utils import get_response  #, judgement
import re
from pprint import pprint
#sleep_time = 20
SHOW = True


def post_process_value(generate_answer, location=-1):
    generate_answer = generate_answer.replace(',', '')                                                  
    generate_answer = ''.join(char for char in generate_answer if not char.isalpha())                   
    generate_answer = ''.join(char for char in generate_answer if char not in ['(', ')'])               
    generate_answer = generate_answer.strip()                                                           
    if type(generate_answer) == str and len(generate_answer) >= 1 and generate_answer[-1] == '.':       
        generate_answer = generate_answer[:-1]
    generate_answer = generate_answer.strip()
    if ' ' in generate_answer:                                                                          
        generate_answer = generate_answer.split(' ')[location]
    if type(generate_answer) == str and len(generate_answer) >= 1:                                      
        pass
    else:
        generate_answer = 0
    if generate_answer in ['-', '=', '+']:                                                              
        generate_answer = 0
    if type(generate_answer) == str and '%' in generate_answer:
        try:
            generate_answer = float(generate_answer.rstrip('%')) / 100
        except SyntaxError as e:
            print(f'SyntaxError during eval: {e}')
            generate_answer = 0
        except Exception as e:
            print(f'Error during eval: {e}')
            generate_answer = 0                                          
        
    if type(generate_answer) == str and ':' in generate_answer:                                          
        generate_answer = generate_answer.replace(':', '.')
    if type(generate_answer) == str and len(generate_answer) >= 1 and generate_answer[-1] in ['.', '/']: 
        generate_answer = generate_answer[:-1]
    if type(generate_answer) == str:
        generate_answer = generate_answer.replace('</>', '') 
        generate_answer = generate_answer.replace('$', '') 
        generate_answer = generate_answer.replace('<>', '').replace('=', '') 
        if len(generate_answer)==0 or generate_answer=='.':
            generate_answer = '0'
        if generate_answer[-1]=='.':
            generate_answer = generate_answer[:-1]
        if len(generate_answer)>=2 and generate_answer[0]=='0':
            generate_answer = generate_answer[1:]
        
        print(f'generate_answer before eval: {generate_answer}')
        
        try:
            generate_answer = eval(generate_answer)
        except SyntaxError as e:
            print(f'SyntaxError during eval: {e}')
            generate_answer = 0
        except Exception as e:
            print(f'Error during eval: {e}')
            generate_answer = 0
        
    return generate_answer


def get_arabic_number(problem, reasoning_path, model, max_length, proxies):
    prompt = f"""
                Q: {problem} 
                A: {reasoning_path} 
                Therefore, the answer (expressed in Arabic numerals and without units) is:
              """
    value = get_response(
        prompt=prompt,
        model=model,
        max_length=max_length, 
        proxies=proxies
    )
    #time.sleep(sleep_time)
    value = post_process_value(value)
    return value


def get_arabic_number_verify(problem, reasoning_path, model, max_length, proxies):
    prompt = f"""
                Q: {problem} 
                A: {reasoning_path} 
                Therefore, X (expressed in Arabic numerals and without units) is:
              """
    value = get_response(
        prompt=prompt,
        model=model,
        max_length=max_length, 
        proxies=proxies
    )
    #time.sleep(sleep_time)
    value = post_process_value(value)
    return value

# def decompose(model, max_length, problem, process_record, proxies):
#     prompt = f'''
#     For the following question, you need to decompose it into several conditions and a final question, in json format. And you should only output the json.
#     An example is shown below:
#     ----
#     <<Input>> :  Sara has 31 red and 15 green balloons. Sandy has 24 red balloons. How many red balloons do they have in total?
#     <<Output>>:
#     {{"conditions": [
#                 "Sara has 31 red balloons,",
#                 "and 15 green balloons.",
#                 "Sandy has 24 red balloons."
#             ],
#     "question": "How many red balloons do they have in total?"
#     }}
#     ----
#     <<Input>>: {problem}
#     <<Output>>:
#     '''

#     response = get_response(
#         prompt=prompt,
#         model=model,
#         max_length=max_length, 
#         proxies=proxies
#     )
#     try:
#         data=json.loads(response)
#         return data
#     except:
#         print("condition get error")
#         return None

# def extract_first_number_sequence(input_string):
#     match = re.search(r'\d+', input_string)
#     print(match)
#     if match:
#         return match.group()
#     return None

# def construct_verify_problem(problem):
#     print(problem)
#     result =extract_first_number_sequence(problem)
#     question=problem.replace(result,"X",1)
#     return question,result

def construct_verify_problem(model, max_length, problem, process_record, proxies):
    prompt = f'''
    Here are two examples:
    --------
    Input : "The serum will make the drinker grow an extra arm every three days and an extra leg every five days."
    Output: three 

    Input: "He creates a movie for $2000. Each DVD cost $6 to make.
    Output: 2000
    --------
    I want you to find the first number (in digit or words) in the input question, and only ouput the number you found.
    Pay attention: you must only output the number same in the origin (like <six> or <6>) in a single number or word.
    Do not output anything else like code.

    Input : "{problem}"
    Output: 
    '''
    #pprint(prompt)
    response = get_response(
        prompt=prompt,
        model=model,
        max_length=max_length, 
        proxies=proxies
    )

    #pprint(result)
    question=problem.replace(response,"X",1)
    return question,response

    # numbers = re.findall(r'\d+',  problem)
    # if numbers:
    #     number = numbers[0]
    #     # Replace the first number with 'X'
    #     output_string = problem.replace(number, 'X')
        
    #     return output_string, number
    # else:
    #     return None

    


    
def initialization(model, max_length, problem, process_record, proxies):
    prompt = f"""
                Q: {problem} After generating the answer, you should add '##answer:' and the final numerical answer at the end (digit only)
                A: Let's think step by step.
            """
    reasoning_path = get_response(
        prompt=prompt,
        model=model,
        max_length=max_length, 
        proxies=proxies
    )

    # json_question=decompose(model, max_length, problem, process_record, proxies)

    # if json_question is None:
    #     json_question=decompose(model, max_length, problem, process_record, proxies)
    
    # if SHOW:
    #     print(f'Initialization Reasoning Path: {reasoning_path}')
    #time.sleep(sleep_time)
    initial_answer = get_arabic_number(problem, reasoning_path, model, max_length, proxies)
    # if SHOW:
    #     print(f'Initialization Numerical Answer: {initial_answer}')
    #time.sleep(sleep_time)
    process_record['Initial_Step'] = {}
    process_record['Initial_Step']['Reasoning'] = reasoning_path
    process_record['Initial_Step']['Answer'] = initial_answer

    return initial_answer,reasoning_path

def checkab(a,b,model, max_length, iter_number, process_record, proxies):
    prompt=f'''
    I want you to decide whether two inputs are the same number, whether they are in words or in digit.
    
    Here are three examples:
    --------
    Input : "three" and "3"
    Output: True

    Input : "three" and "five"
    Output: False

    Input: "3" and "3"
    Ouput: True

    Input : "three" and "5"
    Output: False

    --------
    Pay attention: you must only output True or False. You must not output anything else like code.
    Input : "{a}" and "{b}"
    Output:
    '''

    result = get_response(
        prompt=prompt,
        model=model,
        max_length=max_length, 
        proxies=proxies,
    )

    return result


def verification(generated_answer, verify_problem, verified_answer, model, max_length, iter_number, process_record, proxies):
    prompt = f"""
                Q: {verify_problem} If we know the answer to the above question is {generated_answer}, what is the value of unknown variable X? (If X is irrelevant to the calculation
 process please answer Unknown).
                A: Let's think step by step.
            """
    reasoning_path = get_response(
        prompt=prompt,
        model=model,
        max_length=max_length, 
        proxies=proxies
    )
    if SHOW:
        print(f'Verified Reasoning Path: {reasoning_path}')
    #time.sleep(sleep_time)
    pred_condition = get_arabic_number_verify(verify_problem, reasoning_path, model, max_length, proxies)
    result= checkab(pred_condition,verified_answer, model, max_length, iter_number, process_record, proxies)
    if SHOW:
        print(f'predicted answer : {verified_answer}')
        print(f'Verified Numerical Answer: {pred_condition}')
        print(f"verification result {result == 'True'}")
    #time.sleep(sleep_time)
    process_record[f'Loop_{iter_number}']['verify_reasoning'] = reasoning_path
    process_record[f'Loop_{iter_number}']['verify_answer'] = pred_condition
    
    return (result == 'True') 


def rectification(problem, incorrect_answer, model, max_length, iter_number, process_record, proxies):
    prompt = f"""
                Q: {problem} (The answer is likely not {', '.join(str(x) for x in incorrect_answer)})After generating the answer, you should add '##answer:' and the final numerical answer at the end (digit only).
                A: Let's think step by step.
              """
    reasoning_path = get_response(
        prompt=prompt,
        model=model,
        max_length=max_length, 
        proxies=proxies,
    )
    if SHOW:
        print(f'Rectified Reasoning Path: {reasoning_path}')
    #time.sleep(sleep_time)
    rectified_answer = get_arabic_number(problem, reasoning_path, model, max_length, proxies)
    if SHOW:
        print(f'Rectified Numeirical Answer: {rectified_answer}')
    #time.sleep(sleep_time)
    process_record[f'Loop_{iter_number}']['rectify_reasoning'] = reasoning_path
    process_record[f'Loop_{iter_number}']['rectify_answer'] = rectified_answer
    return rectified_answer, reasoning_path 



def iteration(problem, generated_answer, verify_problem, verified_answer, num_iteration, model, max_length, process_record, proxies):
    incorrect_answer = []
    for iter_number in range(num_iteration):
        process_record[f'Loop_{iter_number}'] = {}
        verification_result = verification(generated_answer[-1], verify_problem, verified_answer,model, max_length, iter_number, process_record, proxies)
        
        if verification_result==True or  (len(generated_answer)>=2 and generated_answer[-1]==generated_answer[-2]):
            process_record['Verify'] = f"True_{iter_number+1}"
            break
        else:
            incorrect_answer.append(generated_answer[-1])
            rectified_answer, full_answer = rectification(problem, incorrect_answer, model, max_length, iter_number, process_record, proxies)
            generated_answer.append(rectified_answer)
            process_record['Verify'] = f"False_{iter_number+1}"
    return generated_answer[-1], full_answer


def pipline(process_record, problem, num_iteration, model, max_length, proxies):
    generated_answer = []
    final_answer =None
    initial_answer, full_answer= initialization(model, max_length, problem, process_record, proxies)
    generated_answer.append(initial_answer)
    
    verify_problem, verified_answer=construct_verify_problem(model, max_length, problem, process_record, proxies)
    try:
        final_answer, full_answer = iteration(problem, generated_answer, verify_problem, verified_answer, num_iteration, model, max_length, process_record, proxies)
    except:
        pass
    return final_answer, full_answer, process_record
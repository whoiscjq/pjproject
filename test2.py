import pandas as pd
import json

def prompt_tmp1(question):

    prompt=f'''
    For the following question, you need to decompose it into several conditions and a final question, in json format. 
    A example it shown below:
    ----
    <<Input>> :  "Sara has 31 red and 15 green balloons . Sandy has 24 red balloons . How many red balloons do they have in total ? "
    <<Output>>:
    {"conditions": [
                "Sara has 31 red,",
                "and 15 green balloons .",
                "Sandy has 24 red balloons ."
            ],
    "question": "How many red balloons do they have in total ?"
    } 
    ----
    <<Input>>:  {question}
    <<Output>>:
    '''
    return prompt
import json
from bot_pipeline import BoT
import argparse
import os
import datetime
from logsetting import logger
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--task_name',type=str,default='gameof24',choices=['gameof24','checkmate','wordsorting'])
parser.add_argument('--api_key',default=None,type=str,help='input your api key here')
parser.add_argument('--model_id',type=str,default='gpt-4o',help='Input model id here, if use local model, input the path to the local model')
parser.add_argument('--rag_dir',type=str,default='./rag_dir',help='Input RAG directory here')



GameOf24 = """
Let's play a game called 24. You will get four numbers. Use each number exactly once, in any order, and only these operations: + – * /. You may add parentheses. Make an expression that equals 24. You only need find one feasible solution.

Example:
Input: 4 9 10 13  
Output: (10 - 4) * (13 - 9) = 24

Input:
"""
CheckmateInOne = """
Given a series of chess moves written in Standard Algebraic Notation (SAN), determine the next move that will result in a checkmate.
Input: 
"""
WordSorting = """
Sort a list of words alphabetically, placing them in a single line of text separated by spaces.
Input:
"""


if __name__ == "__main__":
    args = parser.parse_args()
    task = args.task_name
    api_key = args.api_key
    model_id = args.model_id
    rag_dir = args.rag_dir
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d-%H:%M:%S")
    output_dir = 'test_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    benchmark_dict = {
        'gameof24':GameOf24,
        'checkmate':CheckmateInOne,
        'wordsorting':WordSorting
    }
    
    path_dict = {
        'gameof24':'benchmarks/gameof24.jsonl',
        'checkmate':'benchmarks/CheckmateInOne.jsonl',
        'wordsorting':'benchmarks/word_sorting.jsonl'
    }
    
    buffer_dict = {
        'gameof24':0,
        'checkmate':1,
        'wordsorting':2
        
    }
    
    user_prompt = benchmark_dict[task]
    path = path_dict[task]    
    problem_id = buffer_dict[task]
    test_bot = BoT(
            user_input = None,
            problem_id = problem_id,
            api_key = api_key,
            model_id = model_id,
            need_check = True,
            rag_dir = rag_dir
        )
    
    iterator = enumerate((open(path)), start=1)
    with open(path, 'r', encoding='utf-8') as f: total_lines = sum(1 for _ in f)
    
    for idx, line in tqdm(iterator, total=total_lines):
        if idx < 1: # 1번째부터 시작
            continue
        input = json.loads(line)['input']
        user_input = user_prompt + input
        test_bot.update_input(user_input)
        result = test_bot.bot_run()
        tmp = {'input':input,'result':result}
        with open(f'test_results/BoT_{task}_Llama-3-8B_{timestamp_str}.jsonl', 'a+', encoding='utf-8') as file:
            json_str = json.dumps(tmp)
            file.write(json_str + '\n')
        logger.info("A Problem is completed!")

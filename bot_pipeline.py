import transformers
import torch
import http.client
import json
from accelerate import infer_auto_device_map
from meta_buffer_utilis import meta_distiller_prompt,extract_and_execute_code
from test_templates import game24,checkmate,word_sorting
from meta_buffer import MetaBuffer
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Dict
from transformers import AutoTokenizer
from lightrag import LightRAG, QueryParam
from logsetting import logger

class Pipeline:
    def __init__(self,model_id,api_key=None,base_url='https://api.openai.com/v1/'):
        self.api = False
        self.local = False
        self.base_url = base_url
        self.model_id = model_id
        if api_key == None and model_id != 'gpt-4o':
            self.local = True
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map = 'auto'
            )
        elif api_key != None:
            self.api = True
            self.api_key = api_key
        else:
            logger.critical("Neither API key nor model name were given!")
        self.gen_config_map = {
            "summary": transformers.GenerationConfig(
                do_sample=False,
                num_beams=5,
                max_new_tokens=512,
                repetition_penalty=1.1,
                no_repeat_ngram_size=4,
            ),
            "retrieve": transformers.GenerationConfig(
                do_sample=False,
                max_new_tokens=2048,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
            ),
            "instantiation": transformers.GenerationConfig(
                do_sample=True,
                temperature=0.7,
                top_p=1,
                max_new_tokens=2048,
            ),
            "self-correction": transformers.GenerationConfig(
                do_sample=False,
                num_beams=4,
                early_stopping=True,
                max_new_tokens=128,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            ),
        }
        for profile, cfg in self.gen_config_map.items():
            logger.info(f"decoding_profile={profile} gen_config={cfg}")

    def get_respond(self, meta_prompt, user_prompt, decoding_profile=None):
        if self.api:
            client = OpenAI(api_key=self.api_key,base_url= self.base_url)
            completion = client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": meta_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            response = completion.choices[0].message.content
            return response
        else:
            messages = [
                {"role": "system", "content": meta_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            prompt = self.pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
            )
            
            terminators = [
                self.pipeline.tokenizer.eos_token_id,
                self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            if decoding_profile not in self.gen_config_map:
                raise ValueError(f"No valid decoding_profile: {decoding_profile}!")
            gen_config = self.gen_config_map[decoding_profile]
            
            outputs = self.pipeline(
                prompt,
                generation_config=gen_config
            )
            respond = outputs[0]["generated_text"][len(prompt):]
            
            return respond
            


        
class BoT:
    def __init__(self, user_input,problem_id=0,api_key=None,model_id='gpt-4o',embedding_model='text-embedding-3-large',need_check=False,base_url='https://api.openai.com/v1/',rag_dir=None):
        self.api_key = api_key
        self.model_id = model_id
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.pipeline = Pipeline(self.model_id,self.api_key,self.base_url)
        self.rag_dir = rag_dir or "./rag_dir"
        
        if api_key == None and model_id != 'gpt-4o': # for local model
            self.meta_buffer = MetaBuffer(
                self.model_id,
                None, # local embedding model will be initialized
                api_key='',
                base_url=self.base_url or None,
                rag_dir=self.rag_dir,
            )
            
            # Override llm_model_func into local pipeline
            async def local_llm_async(
                prompt: str,
                system_prompt: Optional[str] = None,
                history_messages: Optional[List[Dict]] = None,
                **kwargs
            ) -> str:
                if history_messages is None:
                    history_messages = []
                    
                meta = system_prompt or ""
                user = prompt
                
                response = self.pipeline.get_respond(meta, user,decoding_profile="retrieve")
                return response

            # Override both MetaBuffer, RAG into local llms
            self.meta_buffer.llm_model_func     = local_llm_async
            self.meta_buffer.rag.llm_model_func = local_llm_async
        elif api_key != None and model_id == 'gpt-4o': # for OpenAI model
            self.meta_buffer = MetaBuffer(
                self.model_id, # gpt-4o
                self.embedding_model,
                api_key=self.api_key,
                base_url=self.base_url or None,
                rag_dir=self.rag_dir,
            )
        else:
            logger.critical("Neither API key nor model name were given!")
            
        
        self.user_input = user_input
        
        # Only for test use, stay tuned for our update
        self.problem_id = problem_id 
        self.need_check = need_check
        
        # test thought templates 추가
        # for template in [game24, checkmate, word_sorting]:
            # self.meta_buffer.rag.insert(template)
        
        # with open("./math.txt") as f:
            # self.meta_buffer.rag.insert(f.read())
            
    def update_input(self,new_input):
        self.user_input = new_input
        
    def problem_distillation(self):
        logger.info(f"User prompt: {self.user_input}")
        self.distilled_information = self.pipeline.get_respond(meta_distiller_prompt, self.user_input, decoding_profile="summary")
        logger.info(f'Distilled information:{self.distilled_information}')

    def buffer_retrieve(self):
        search_query = "Find the most similar thought-template in the database for this information:\n"+self.distilled_information
        logger.debug(f"RAG search query: {search_query}")
        self.thought_template = self.meta_buffer.rag.query(search_query, param=QueryParam(
                mode="naive",
                only_need_context=False,
            )
        )
        logger.info(f'Retrieved thought template(RAG response): {self.thought_template}')
        
    def buffer_instantiation(self):
        self.instantiation_instruct = """
You are an expert in problem analysis and can apply previous problem-solving approaches to new issues. The user will provide a specific task description and a thought template. Your goal is to analyze the user's task and generate a specific solution based on the thought template. If the instantiated solution involves Python code, only provide the code and let the compiler handle it. If the solution does not involve code, provide a final answer that is easy to extract from the text.
It should be noted that all the python code should be within one code block, the answer should not include more than one code block! And strictly follow the thought-template to instantiate the python code but you should also adjust the input parameter according to the user input!
"""

        self.formated_input = f"""
Distilled information:
{self.distilled_information}
User Input:
{self.user_input}
Thought template:
{self.thought_template}

Instantiated Solution:
Please analyze the above user task description and thought template, and generate a specific, detailed solution. If the solution involves Python code, only provide the code. If not, provide a clear and extractable final answer.        
"""
        self.result = self.pipeline.get_respond(self.instantiation_instruct,self.formated_input, decoding_profile="instantiation")
        logger.info(f'Instantiated reasoning result: {self.result}')
        
    def buffer_manager(self):
        self.problem_solution_pair = self.user_input + self.result
        self.thought_distillation()
        self.meta_buffer.dynamic_update(self.pipeline, self.distilled_thought)
        
    def thought_distillation(self):
        thought_distillation_prompt = """You are an expert in problem analysis and generalization. Your task is to follow the format of thought template below and distill a high-level thought template to solve similar problems:
        Example thought template:
        ### Problem Type 20: Solution Concentration Problem

**Definition**: This type of problem involves the relationship between a solvent (water or another liquid), solute, solution, and concentration.

**Quantitative Relationships**:
- Solution = Solvent + Solute
- Concentration = Solute ÷ Solution × 100%

**Solution Strategy**: Use the formulas and their variations to analyze and calculate the problem.

**Example**: There is 50 grams of a 16% sugar solution. How much water needs to be added to dilute it to a 10% sugar solution?

**Solution**:
Using the formula:  
50 × 16% ÷ 10% - 50 = 30 grams of water need to be added.

It should be noted that you should only return the thought template without any extra output.
        """
        self.distilled_thought = self.pipeline.get_respond(thought_distillation_prompt, self.problem_solution_pair, decoding_profile="summary")
        logger.info(f'Distilled thought: {self.distilled_thought}')
    def reasoner_instantiation(self):
        # Temporay using selection method to select answer extract method
        
        problem_id_list = [0,1,2]
        self.instantiation_instruct = """
You are an expert in problem analysis and can apply previous problem-solving approaches to new issues. The user will provide a specific task description and a thought template. Your goal is to analyze the user's task and generate a specific solution based on the thought template. If the instantiated solution involves Python code, only provide the code and let the compiler handle it. If the solution does not involve code, provide a final answer that is easy to extract from the text.
It should be noted that all the python code should be within one code block, the answer should not include more than one code block! And strictly follow the thought-template to instantiate the python code but you should also adjust the input parameter according to the user input!
        """
        
        self.formated_input = f"""
Distilled information:
{self.distilled_information}
User Input:
{self.user_input}
Thought template:
{self.thought_template}

Instantiated Solution:
Please analyze the above user task description and thought template, and generate a specific, detailed solution. If the solution involves Python code, only provide the code. If not, provide a clear and extractable final answer.        
        """
        self.inspector_prompt = """
You are an excellent python programming master who are proficient in analyzing and editing python code, and you are also good at understanding the real-world problem. Your task is:
1. Analyze the given python code
2. Edit the input code to make sure the edited code is correct and could run and solve the problem correctly.  
Your respond **should follow the format** below:
```python
## Edited code here
```
        """
        self.result = self.pipeline.get_respond(self.instantiation_instruct,self.formated_input, decoding_profile="instantiation")
        logger.info(f'(1st) Instantiated reasoning result: {self.result}')
        if self.problem_id in problem_id_list:
            self.final_result, code_str = extract_and_execute_code(self.result)
            logger.debug(f"self.final_result: {self.final_result}")
            logger.debug(f"code_str: {code_str}")
            if self.need_check:
                self.count = 0
                self.inter_input = f"""
                User_input:{self.user_input}
                {code_str}
                {self.final_result}
                """
                self.inter_result = self.final_result
                while(('An error occurred' in self.inter_result) or (self.inter_result == '') or (self.inter_result == 'None')):
                    logger.warning(f'The code cannot be executed correctly, here we continue the edit phase: {self.inter_result}')
                    logger.warning(f'The problem code is: {code_str}')
                    self.inter_input = self.pipeline.get_respond(self.inspector_prompt,self.inter_input, decoding_profile="self-correction")
                    print(self.inter_input)
                    self.inter_result, inter_code_str = extract_and_execute_code(self.inter_input)
                    self.inter_input = f"""
                User_input:{self.user_input}
                {inter_code_str}
                The result of code execution: {self.inter_result}
                """
                    self.count = self.count + 1
                    if self.count > 3:
                        break
                self.final_result = self.inter_result 
            logger.info(f'({self.count+1}th) The result of code execution: {self.final_result}')
        else:
            self.final_result = self.result

    
    def bot_run(self):
        self.problem_distillation()
        self.buffer_retrieve()
        # self.reasoner_instantiation()
        self.buffer_instantiation()
        self.buffer_manager()
        
        # return self.final_result # for reasoner_instantiation() case
        return self.result # for buffer_instantiation() case
    
    def bot_inference(self):
        self.problem_distillation()
        self.buffer_retrieve()
        self.buffer_instantiation()
        # self.buffer_manager()
        # logger.info(f'Final results: {self.result}')
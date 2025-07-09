import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete,openai_complete_if_cache,openai_embedding,hf_model_complete
import numpy as np
from lightrag.utils import EmbeddingFunc, compute_args_hash
import asyncio

class MetaBuffer:
    def __init__(self, llm_model, embedding_model, api_key=None,
                 base_url="https://api.openai.com/v1/", rag_dir='./rag_dir'):
        self.api_key = api_key
        self.llm = llm_model
        self.base_url = base_url
        if callable(embedding_model):
            self.embedding_func = embedding_model
            try:
                embedding_dim = embedding_model.__self__.get_sentence_embedding_dimension()
            except Exception:
                embedding_dim = 384
        else:
            self.embedding_model = embedding_model
            async def _openai_embed(texts: list[str]) -> np.ndarray:
                return await openai_embedding(
                    texts,
                    model=self.embedding_model,
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            self.embedding_func = _openai_embed
            embedding_dim = 3072
        if not os.path.exists(rag_dir):
            os.makedirs(rag_dir, exist_ok=True)
        self.rag = LightRAG(
            working_dir=rag_dir,
            llm_model_func=self.llm_model_func,
            llm_model_name=self.llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                func=self.embedding_func
            )
        )
        
    async def llm_model_func(
        self, prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        return await openai_complete_if_cache(
            self.llm,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs
        )
    async def embedding_func(self, texts: list[str]) -> np.ndarray:
        
        return await openai_embedding(
            texts,
            model= self.embedding_model,
            api_key= self.api_key,
            base_url= self.base_url
        )
    
    def retrieve_and_instantiate(self,search_query,run_prompt=None):
        # retrieve
        ctx = self.rag.query(search_query, param=QueryParam(
                mode="hybrid",
                only_need_context=True,
            )
        )
        print(f"[retrieve_and_instantiate] {{retrieve result}} {ctx}")
        
        # instantiation when prompt is not empty
        if run_prompt != None:
            full_prompt = ctx + "\n" + run_prompt
            response = asyncio.run(self.llm_model_func(full_prompt))
            return response
            
        return ctx
    
    def dynamic_update(self, thought_template):
        decision_prompt = """
Now Find most relevant thought template in the MetaBuffer according to the given thought template, and Determine whether there is a fundamental difference in the problem-solving approach between this and the most similar thought template in MetaBuffer. If there is  fundamental difference, output "True." Otherwise, output "False." Answer with only True or False.
"""

        # RAG에서 context만 추출
        ctx = self.rag.query(
            thought_template,
            param=QueryParam(
                mode="hybrid",
                only_need_context=True
            )
        )

        # LLM에 최종 판단 요청
        full_prompt = ctx + "\n" + decision_prompt + thought_template
        response = asyncio.run(self.llm_model_func(full_prompt))

        print("[dynamic_update] raw LLM response:", response)

        if self.extract_similarity_decision(response):
            print("[dynamic_update] MetaBuffer Updated!")
            self.rag.insert(thought_template)
        else:
            print("[dynamic_update] No need to Update!")

        
    def extract_similarity_decision(self,text):
        """
        This function takes the input text of an example and extracts the final decision
        on whether the templates are similar or not (True or False).
        """
        # Convert the text to lowercase for easier matching
        text = text.lower()
        
        # Look for the conclusion part where the decision is made
        if "false" in text:
            return False
        elif "true" in text:
            return True
        else:
            # In case no valid conclusion is found
            raise ValueError("No valid conclusion (True/False) found in the text.")
        
        
    
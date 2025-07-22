import os
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete,openai_complete_if_cache,openai_embed
from lightrag.llm.hf import hf_model_complete, hf_embed
from lightrag.utils import EmbeddingFunc, compute_args_hash
from lightrag.kg.shared_storage import initialize_pipeline_status
from transformers import AutoModel, AutoTokenizer
import asyncio
from logsetting import logger
import nest_asyncio
from sklearn.metrics.pairwise import cosine_similarity


nest_asyncio.apply()
LOOP = asyncio.get_event_loop()

class MetaBuffer:
    def __init__(self, llm_model, embedding_model, api_key=None, base_url="https://api.openai.com/v1/", rag_dir='./rag_dir'):
        self.api_key = api_key
        self.llm_name = llm_model
        self.base_url = base_url
        self.DELTA = 0.7
        if not os.path.exists(rag_dir):
            os.makedirs(rag_dir, exist_ok=True)
        if api_key == '':
            embedding_func=EmbeddingFunc(
                    embedding_dim=1024,
                    # embedding_dim=384,
                    max_token_size=8192,
                    func=lambda texts: hf_embed(
                        texts,
                        tokenizer=AutoTokenizer.from_pretrained(
                            "BAAI/bge-m3"
                            # "sentence-transformers/all-MiniLM-L6-v2"
                        ),
                        embed_model=AutoModel.from_pretrained(
                            "BAAI/bge-m3"
                            # "sentence-transformers/all-MiniLM-L6-v2"
                        ),
                    ),
                )
        else:
            embedding_func=openai_embed
        async def initialize_rag(WORKING_DIR, llm_model):
            # To use OpenAI model, you should run `export OPENAI_API_KEY="sk-…"` in your terminal, it will be modified later.
            rag = LightRAG(
                working_dir=WORKING_DIR,
                llm_model_func=gpt_4o_complete,
                llm_model_name=llm_model,
                # llm_model_name="meta-llama/Llama-3.1-8B-Instruct",
                embedding_func=embedding_func,
            )

            await rag.initialize_storages()
            await initialize_pipeline_status()

            return rag
        self.rag = asyncio.run(initialize_rag(rag_dir, llm_model))
    async def llm_model_func(
        self, prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        return await openai_complete_if_cache(
            self.llm_name,
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
                mode="naive",
                only_need_context=False,
            )
        )
        logger.debug(f"A type of ctx: {type(ctx)}")
        logger.debug(f"Raw data of ctx(retrieved result): {ctx}")
        
        # instantiation when prompt is not empty
        if run_prompt != None:
            full_prompt = run_prompt + "\n" + ctx
            
            response = asyncio.run(self.llm_model_func(full_prompt)) # for local model
            # response = LOOP.run_until_complete(self.llm_model_func(full_prompt)) # for OpenAI model
            return response
            
        return ctx
    
    def is_new_template(self,
        proposed_template: str,
        existing_template: str,
        threshold: float
    ) -> bool:
        vecs = asyncio.run(
            self.rag.embedding_func(
                [proposed_template, existing_template]
            )
        )
        sim = cosine_similarity([vecs[0]], [vecs[1]])[0][0]
        
        return (sim < threshold)
        
    
    def dynamic_update(self, pipeline, thought_template):
        search_query = "Find the most similar thought-template in the database for this thought template:\n"+thought_template
        logger.debug(f"RAG search query: {search_query}")
        ctx = self.rag.query(
            search_query,
            param=QueryParam(
                mode="naive",
                only_need_context=False,
            )
        )
        logger.debug(f"The most relevant thought-template in RAG DB (RAG response): {ctx}")
        
        # 1차 필터링: threshold updating rule
        # if ctx and self.is_new_template(thought_template, ctx, self.DELTA):
            # logger.debug("It is new template! Temporally we insert this thought template.")
            # self.rag.insert(thought_template)
            # return
        # else:
            # logger.debug("It is not a new template! Now we terminates")
            # return
        
        # 2차 필터링: LLM에게 판단 요청
        system_prompt = """
Now we found the most relevant thought template in the MetaBuffer according to the given thought template. Determine whether there is a fundamental difference in the problem-solving approach between given template and the most similar template in MetaBuffer. If there is a fundamental difference, output "True". Otherwise, output "False". Respond **only** “True” or “False” (case‑sensitive) with no other text.
"""
        meta_prompt = """
# Our Thought template
{thought_template}
# The most relevant Thought template
{ctx}
"""
        user_prompt = meta_prompt.format(thought_template=thought_template, ctx=ctx)
        logger.debug(f"user_prompt: {user_prompt}")

        response = pipeline.get_respond(system_prompt, user_prompt, decoding_profile="instantiation") # for local model
        # response = LOOP.run_until_complete(self.llm_model_func(system_prompt+"\n"+user_prompt)) # for OpenAI model

        logger.info(f"raw LLM response (True or False): {response}")
        
        if self.extract_similarity_decision(response):
            logger.info("MetaBuffer Updated!")
            self.rag.insert(thought_template)
        else:
            logger.info("No need to Update!")

        
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
        
        
    
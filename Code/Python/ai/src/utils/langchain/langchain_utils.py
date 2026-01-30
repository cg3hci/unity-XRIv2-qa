from settings import safe_enable_openai_key, LANGCHAIN_VERSION, GPT4ALL_MODELS_PATH, safe_enable_huggingface_token
safe_enable_openai_key()
safe_enable_huggingface_token()

from typing import TypedDict
from src.utils.os_utils import dictionary_to_filename
import ast
import os


########### COMMON ###########
from langchain_core.language_models.llms import LLM
#############################

########### VERSION SPECIFIC ###########
if LANGCHAIN_VERSION=='0.0.35*':
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.chains.llm import LLMChain
    from langchain.chains import RetrievalQA
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate

    from langchain.chat_models import ChatOpenAI
    # GPT4ALL is not supported in this version

    # import pinecone
    from pinecone import PineconeProtocolError
elif LANGCHAIN_VERSION=='0.3.*':
    from langchain_openai import OpenAIEmbeddings
    # nothing to import Ffor LLMChain, RetrievalQA
    from langchain_core.runnables import RunnablePassthrough # but you need this runnable for retrievalQA
    # ConversationBufferMemory is not supported in this version, need to understand how to replace it
    from langchain_core.prompts import PromptTemplate

    # HUGGINGFACE
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError
    # OPENAI
    from langchain_openai import ChatOpenAI, OpenAI
    # GPT4ALL
    from langchain_community.llms.gpt4all import GPT4All
    # GOOGLE GENERATIVE AI
    from langchain_google_genai import ChatGoogleGenerativeAI

    # PINECONE
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore

    # from langchain.chains import create_retrieval_chain
    # from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.output_parsers import StrOutputParser

    from langchain_core.messages.ai import AIMessage

else :
    raise ValueError("Invalid LANGCHAIN_VERSION")
#############################

class TemplateHelper():
    @staticmethod
    def generate_prompt_template(input_variables:list[str], template:str):
        return PromptTemplate(input_variables=input_variables, template=template)


class EmbeddingHelper():  
    def __init__(self, model_helper:'ModelNameHelper.Embedding.EmbeddingInfo'):
        VALID_MODELS = [ModelNameHelper.Embedding.LARGE(), ModelNameHelper.Embedding.SMALL(), ModelNameHelper.Embedding.OLD_ADA()]
        if not (model_helper in VALID_MODELS):
            raise ValueError(f"Invalid model name: {model_helper}. Valid models are: {VALID_MODELS}")
        
        self.model_helper = model_helper
        self.embed = OpenAIEmbeddings(model=model_helper["name"])

    def embed_query(self, query:str)->list[float]:
        return self.embed.embed_query(query)
    
    def get_output_dim(self)->int:
        return self.model_helper["output_dim"]
    
    def get_embedding_model(self)->OpenAIEmbeddings:
        return self.embed

class LLMChainHelper():
    def __init__(self, m_llm:LLM, m_prompt:PromptTemplate, want_memory:bool=False):       
        if LANGCHAIN_VERSION=='0.0.35*':
            if want_memory:
                mem_object = ConversationBufferMemory(memory_key="chat_history")
            else:
                mem_object = None
            self.llm_chain = LLMChain(llm=m_llm, prompt=m_prompt, memory=mem_object)

        elif LANGCHAIN_VERSION=='0.3.*':
            if want_memory:
                raise ValueError("This library does not support the langchain memory system in the current version.")
            self.llm_chain = m_prompt | m_llm
        

    def invoke(self, question:str)->str:
        if not isinstance(question, str):
            raise ValueError(f"Invalid input type: {type(question)}")

        if LANGCHAIN_VERSION=='0.0.35*':
            return self.llm_chain.invoke({"question": question})
        elif LANGCHAIN_VERSION=='0.3.*':
            output_chain = self.llm_chain.invoke({"question":question})
            if isinstance(output_chain, dict):
                if "content" in output_chain:
                    output_chain = output_chain["content"]
                else:
                    raise ValueError(f"Invalid output type: {type(output_chain)}")
            elif isinstance(output_chain, AIMessage):
                output_chain = output_chain.content
            elif not isinstance(output_chain, str):
                raise ValueError(f"Invalid output type: {type(output_chain)}")
            return {"question":question, "text":output_chain}
    

class RetrievalQAChainHelper():
    def __init__(self, m_llm:LLM, m_retriever,m_prompt:PromptTemplate, want_memory:bool=False, m_chain_type:str=''):
        if m_chain_type=='':
            m_chain_type = ChainArgsHelper.AllowedChainType.STUFF   

        if LANGCHAIN_VERSION=='0.0.35*':
            m_chain_type_kwargs = {"prompt":m_prompt}
            if want_memory:
                m_chain_type_kwargs["memory"] = ConversationBufferMemory(memory_key="chat_history", input_key="question")
                
            self.__qa_chain = RetrievalQA.from_chain_type(
                llm=m_llm,
                retriever=m_retriever,
                chain_type=m_chain_type,
                chain_type_kwargs=m_chain_type_kwargs,
                return_source_documents=True
            )
        elif LANGCHAIN_VERSION=='0.3.*':
            if want_memory:
                raise ValueError("This library does not support the langchain memory system in the current version.")
            
            # METHOD OF START
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)


            # METHOD DOC
            rag_chain_from_docs = (
                {
                    "question": lambda x: x["question"],  # input query
                    "context": lambda x: format_docs(x["context"]),  # context
                }
                | m_prompt  # format query and context into prompt
                | m_llm  # generate response
                | StrOutputParser()  # coerce to string
            )
            retrieve_docs = (lambda x: x["question"]) | m_retriever
            self.__qa_chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
                answer=rag_chain_from_docs
            )


    def invoke(self, question:str)->str:
        if not isinstance(question, str):
            raise ValueError(f"Invalid input type: {type(question)}")
        
        if LANGCHAIN_VERSION=='0.0.35*':
            """Output be like
            {
                "query": <THE USER INPUT/QUESTION>,
                "results": <THE LLM RESPONSE>,
                "source_documents": [ # The documents that were used to generate the response
                    Document( # i-th document
                        "page_content"=<THE CONTEXT/ANSWER OF THE I-TH RETRIEVED DOCUMENT>,
                        "metadata" = {
                            "q_id" = <ID OF THE RETRIEVED DOCUMENT IN THE VECTORSTORE>,
                            "question = <THE QUESTION ASKED TO HAVE THE ANSWER INSIDE "PAGE_CONTENT">
                        }
                    ), ...
                ]
            }
            """
            return self.__qa_chain.invoke({"query": question})
        elif LANGCHAIN_VERSION=='0.3.*':
            output_chain = self.__qa_chain.invoke({"question":question})
            keys_mandatory = ["question", "answer", "context"]
            if not all(key in output_chain for key in keys_mandatory):
                raise ValueError(f"Invalid output chain: {output_chain}. Missing keys: {keys_mandatory}")

            return {"query":question, "result":output_chain["answer"], "source_documents":output_chain["context"]}
    
def AI_stringified_list__to__list(important_props_str:str)->list[str]:
    """Converts a stringified list to a list of strings."""
    out = []
    try:
        out = ast.literal_eval(important_props_str)
    except SyntaxError:
        try:
            def parse_dash_names(input_string: str) -> list:
                # Split the input string by the newline character
                lines = input_string.split('\n')
                
                names = []
                for line in lines:
                    stripped_line = line.strip()
                    
                    # Check if the line starts with a dash followed by an optional space and a valid name
                    if not stripped_line.startswith('-') or len(stripped_line) <= 1:
                        raise ValueError(f"Invalid format: '{line}'")
                    
                    # Extract the name by removing the dash and optional space
                    name = stripped_line[1:].strip()
                    
                    # Check if the name is non-empty after removing the dash
                    if not name or name.isspace():
                        raise ValueError(f"Invalid format: '{line}'")
                    
                    names.append(name)
                
                return names

            out=parse_dash_names(important_props_str)
        except ValueError:
            print(f"Occured an error while doing ast.literal_eval(answer[0]).\nThe extrapolated answer is: {important_props_str[0]}\nThe full answer is {important_props_str}")
    return out

class LLMHelper():
    @staticmethod
    def __is_openai_model_llm(model_name:str)->bool:
        return model_name in [ModelNameHelper.Text2Text.GPT3_5_TURBO(), ModelNameHelper.MultiModal2Text.GPT4(), ModelNameHelper.MultiModal2Text.GPT4_OMNI(), ModelNameHelper.MultiModal2Text.GPT4_OMNI_CHEAP(), ModelNameHelper.MultiModal2Text.GPT4_OMNI_SUPER_CHEAP()]
    
    @staticmethod
    def __is_gpt4all_model(model_name:str)->bool:
        is_file_type_gguf = model_name.lower().endswith(".gguf")
        does_file_exist = os.path.exists(f"{GPT4ALL_MODELS_PATH}{model_name}")
        return is_file_type_gguf and does_file_exist

    @staticmethod
    def __is_huggingface_model(model_id:str)->bool:
        api = HfApi()
        try:
            api.model_info(model_id)
            return True  # Model ID is valid
        except RepositoryNotFoundError:
            return False  # Model ID does not exist
        except Exception as e:
            print(f"An error occurred: {e}")
            return False  # Handle other errors gracefully
        
    @staticmethod
    def __is_google_generative_ai_model(model_name:str)->bool:
        return model_name in [ModelNameHelper.MultiModal2Text.GEMINI_1DOT5_FLASH_SUPERCHEAP(), ModelNameHelper.MultiModal2Text.GEMINI_1DOT5_FLASH(), ModelNameHelper.MultiModal2Text.GEMINI_1DOT5_PRO_EXPENSIVE(), ModelNameHelper.MultiModal2Text.GEMINI_1DOT0_PRO_GARBAGE_QUESTION_MARK()]

    @staticmethod
    def get_model(model_name:str, streaming:bool=False, callbacks:list=None)->LLM:
        if LANGCHAIN_VERSION=='0.0.35*':
            if __class__.__is_openai_model_llm(model_name):
                return ChatOpenAI(model_name=model_name, temperature=0, streaming=streaming, callbacks=callbacks)
                
            raise ValueError(f"Model name {model_name} is not valid.") 
            
        elif LANGCHAIN_VERSION=='0.3.*':            
            if __class__.__is_openai_model_llm(model_name):
                return ChatOpenAI(model_name=model_name, temperature=0, streaming=streaming, callbacks=callbacks)


            if __class__.__is_google_generative_ai_model(model_name):
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=1,
                    # other params...
                )                        
            if __class__.__is_gpt4all_model(model_name):
                return GPT4All(model=f"{GPT4ALL_MODELS_PATH}{model_name}", device="gpu", verbose=True, n_predict=1024*3, max_tokens=1024*13, seed=42, temp=0,n_threads=7, streaming=streaming, callbacks=callbacks)

            if __class__.__is_huggingface_model(model_name):
                llm = HuggingFacePipeline.from_model_id(
                    model_id=model_name,
                    task="text-generation",
                    pipeline_kwargs={
                        "max_new_tokens": 100,
                        "top_k": 50,
                        "temperature": 0,
                    },
                )

                chat_model = ChatHuggingFace(llm=llm) # Do I want this? # https://huggingface.co/blog/langchain#chathuggingface
                return chat_model #llm
            raise ValueError(f"Invalid model name: {model_name}.")

                
        else:
            raise ValueError("Invalid LANGCHAIN_VERSION")

class ModelNameHelper():
    # curl https://api.openai.com/v1/models -H "Authorization: Bearer <YOUR_KEY>"
    class Embedding():
        class EmbeddingInfo(TypedDict):
            name:str
            output_dim:int
        
        @staticmethod
        def LARGE()->EmbeddingInfo: return {"name":"text-embedding-3-large", "output_dim":3072}

        @staticmethod
        def SMALL()->EmbeddingInfo:
            """Less expensive than the large embedding model, but still very powerful. It is the best choice for most applications.
            
            Better than the ada-002 model."""
            return {"name":"text-embedding-3-small", "output_dim":1536}
        
        @staticmethod
        def OLD_ADA()->EmbeddingInfo: return {"name":"text-embedding-ada-002", "output_dim":1536}

    class MultiModal2Image():
        @staticmethod
        def DALLE_2():
            """
            (1) For creating N VARIATIONS of an EXISTING image. NO prompts in INPUT. \n
            (2) For generating variations in an EXISTING IMAGE by specificing the MASK AREA and a TEXT PROMPT
            (3) Like DALL-E 3, can generate images from scratch given a text prompt.
            
            PROS
                <li> Cheaper </li>
            CONS
                <li> Less quality </li>"""        
            return "dall-e-2"

        @staticmethod
        def DALLE_3():
            """ (1) For generating images from scratch given a text promt."""
            return "dall-e-3"

    class Text2Text():
        @staticmethod
        def GPT3_5_TURBO(): return "gpt-3.5-turbo"

        @staticmethod
        def CUSTOM_FINETUNED_GPT3_5_TURBO(): return "ft:gpt-3.5-turbo-0613:personal:mrtk3-bot:8cuqWf88"

        @staticmethod
        def LLAMA_3_dot_2__3B(): return "Llama-3.2-3B-Instruct-Q4_0.gguf"

        @staticmethod
        def LLAMA_3__8B_Context16K(): return "Llama-3-8B-16K.Q4_0.gguf" # https://huggingface.co/QuantFactory/Llama-3-8B-16K-GGUF
        
        @staticmethod
        def QWEN_1DOT5__7B_Chat_q4(): return "codeqwen-1_5-7b-chat-q4_0.gguf"
        
        @staticmethod
        def LLAMA_2__7B_Context4K(): return "llama-2-7b-chat.Q4_0.gguf"

        @staticmethod
        def LLAMA_2__7B_Context32K(): return "llama-2-7b-32k-instruct.Q4_0.gguf"

        @staticmethod
        def PHI_3__3_DOT_8B(): return "microsoft/Phi-3-mini-4k-instruct"

        @staticmethod
        def GEMMA_2_9B_it_q8(): return "gemma-2-9b-it-Q8_0-f16.gguf"

        @staticmethod
        def Mixtral_2x7B_Q4(): return "laser-dolphin-mixtral-2x7b-dpo.Q4_0.gguf"

        @staticmethod
        def Phi_3DOT5_Mini_3DOT8B_Q4(): return "Phi-3.5-mini-instruct-Q4_0.gguf"

        @staticmethod
        def MISTRAL_7B_OPENORCA_Q4(): return "mistral-7b-openorca.Q4_0.gguf"
 
        @staticmethod
        def DEEPSEEK_R1(): return "deepseek-coder-6.7b-instruct.Q4_0.gguf"


    class MultiModal2Text():
        @staticmethod
        def GPT4(): return "gpt-4"
        
        @staticmethod
        def GPT4_OMNI(): return "gpt-4o"

        @staticmethod
        def GPT4_OMNI_CHEAP(): return "gpt-4o-2024-08-06"

        @staticmethod
        def GPT4_OMNI_SUPER_CHEAP(): return "gpt-4o-mini"

        @staticmethod
        def GEMINI_1DOT5_FLASH_SUPERCHEAP(): return "gemini-1.5-flash-8b"

        @staticmethod
        def GEMINI_1DOT5_FLASH(): return "gemini-1.5-flash"

        @staticmethod
        def GEMINI_1DOT5_PRO_EXPENSIVE(): return "gemini-1.5-pro"

        @staticmethod
        def GEMINI_1DOT0_PRO_GARBAGE_QUESTION_MARK(): return "gemini-1.0-pro"

    class Text2Speech():
        @staticmethod
        def TTS_REAL_TIME():
            """Cheaper and faster than the counterpart High Definition model."""
            return "tts-1"
        
        @staticmethod
        def TTS_HD():
            """Higher quality than the counterpart Real Time model."""
            return "tts-1-hd"

    class Speech2Text():
        @staticmethod
        def WHISPER():
            return "whisper-1"

class ChainArgsHelper():
    class AllowedChainType():
        # DEFAULT: Takes the list of input documents, inserts them all into a prompt and passes the prompt to the LLM
        # https://python.langchain.com/docs/modules/chains/document/stuff
        STUFF="stuff" 
        # Loops over the input documents. Each iteration of the loop passes the current document to the LLM along with the output of the previous iteration.
        # https://python.langchain.com/docs/modules/chains/document/refine
        REFINE="refine"  
        # Does N 'stuff' chain passing one of the N input documents (map). All chain responses are grouped (reduce) and treated as input documents for another 'stuff' chain.
        # https://python.langchain.com/docs/modules/chains/document/map_reduce
        MAP_REDUCE="map_reduce" 
        # Does the 'map' step, but each of the N chain responsed has also a rank value. The response with the highest rank is the final response.
        # https://python.langchain.com/docs/modules/chains/document/map_rerank
        MAP_RERANK="map_rerank" 
  
class RetrieverHelper():
    class AllowedSearchType():
        SIMILARITY = "similarity"
        SIMILARITY_SCORE_THRESHOLD = "similarity_score_threshold"
        MMR = "mmr"
    
        def get_default(): return __class__.SIMILARITY

    class SearchKwargs_FromArgs():
        def similarity(k:int = 4, filter:dict = None):
            """ Return docs most similar to query.
            
            Args:
                k is the number of (most similar) documents the Retriever will return
                filter is a dictionary of filters to apply to the results in the form {<META FIELD>: {<STR_OP>: <STR_VALUE>}} e.g. {"q_id": {"$ne": "specific_doc_id"}}
            """
            d = {'k':k}
            if filter is not None: d['filter'] = filter
            return d
        
        def similarity_score_threshold(k:int = 4, score_threshold:float = 0.5, filter:dict = None): 
            """ Return docs and relevance scores in the range [0, 1].

            0 is dissimilar, 1 is most similar.

            Args:
                query: input text
                k: Number of Documents to return. Defaults to 4.
                **kwargs: kwargs to be passed to similarity search. Should include:
                    score_threshold: Optional, a floating point value between 0 to 1 to
                        filter the resulting set of retrieved docs
                filter is a dictionary of filters to apply to the results in the form {<META FIELD>: {<STR_OP>: <STR_VALUE>}} e.g. {"q_id": {"$ne": "specific_doc_id"}}

            Returns:
                List of Tuples of (doc, similarity_score)
            """
            d = {'k':k, 'score_threshold':score_threshold}
            if filter is not None: d['filter'] = filter
            return d

        def mmr(k:int = 4, fetch_k:int = 20, lambda_mul:float = 0.5, filter:dict = None): 
            """[PASTING THE ORIGINAL DOC] Return docs selected using the maximal marginal relevance.

            Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

            Args:
                query: Text to look up documents similar to.
                k: Number of Documents to return.
                fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                lambda_mult: Number between 0 and 1 that determines the degree
                            of diversity among the results with 0 corresponding
                            to maximum diversity and 1 to minimum diversity.
                filter is a dictionary of filters to apply to the results in the form {<META FIELD>: {<STR_OP>: <STR_VALUE>}} e.g. {"q_id": {"$ne": "specific_doc_id"}}

            Returns:
                List of Documents selected by maximal marginal relevance.
            """
            d = {'k':k, 'fetch_k':fetch_k, 'lambda_mul':lambda_mul}
            if filter is not None: d['filter'] = filter
            return d
        
        def stringify(d:dict):
            """Return a string representation of the dictionary"""
            if 'filter' not in d:
                return dictionary_to_filename(d)
            
            copied = d.copy()
            copied['filter'] = 'filter_present'
            return dictionary_to_filename(copied)

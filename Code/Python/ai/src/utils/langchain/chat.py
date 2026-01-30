import json
import os
import time
from abc import ABC, abstractmethod

from pathvalidate import sanitize_filename
from settings import safe_enable_openai_key, safe_disable_openai_key, is_openai_key_enabled
from src.my_prompts import ChatPromptTemplates
from src.utils.langchain.langchain_utils import LLMChainHelper, PromptTemplate, LLMHelper, RetrievalQAChainHelper
from src.utils.langchain.langchain_utils import RetrieverHelper, ChainArgsHelper, ModelNameHelper, VectorStore, \
    Document, BaseCallbackHandler
from src.utils.os_utils import safe_open_w, CONVERSATIONS_PATH, dictionary_to_filename


#region Helpers

class Conversation():

    def __init__(self):
        self.json = {"conversation": []}


    def add_interaction(self, user_question:str, bot_response:str, source_documents:list[Document]=[]):
        self.json["conversation"].append({"user": user_question, "bot": bot_response, "source_documents": source_documents})


    def dump_conversation(self, dest_folder:str):

        def dump_to_json(JSON, dest_folder):
            path = os.path.join(dest_folder, "all.json")

            print("Dumping conversation to JSON file...")
            with safe_open_w(path) as fp:
                json.dump(JSON, fp)
            print("Conversation saved correctly to", path)

        def dump_to_markdown(JSON, dest_folder):
            print("Dumping conversation to MARKDOWN file...")
            
            for idx, conv in enumerate(JSON["conversation"]):
                path = os.path.join(dest_folder, f"{idx}.md")

                with safe_open_w(path) as fp:
                    fp.write(f"## User:\n{conv['user']}\n")
                    fp.write(f"## Bot:\n{conv['bot']}\n")
                    if "source_documents" in conv:
                        fp.write(f"### Source Documents:\n")
                        if not isinstance(conv["source_documents"], list) or len(conv["source_documents"]) == 0:
                            fp.write("None\n")
                        else:
                            for doc in conv["source_documents"]:
                                if not isinstance(doc, dict):
                                    fp.write("Invalid Source Document beucase it's not a dict\n")
                                elif "metadata" not in doc:
                                    fp.write("No tag metadata in the source document\n")
                                elif len(doc['metadata']) == 0:
                                    fp.write("Empty metadata\n")
                                else:
                                    for key, value in doc['metadata'].items():
                                        fp.write(f"#### Metadata key [{key}]\n{value}\n")
                            fp.write("\n")

            print("Conversation saved correctly to", path)

        dump_to_json(self.json, dest_folder)
        dump_to_markdown(self.json, dest_folder)
#endregion
class GenericChatbot(ABC):
    ALLOWED_TO_SPEND=True
    
    def __init__(self, model_name, memory:bool=False, stream:bool=False, callbacks=[]):
        if not self.ALLOWED_TO_SPEND:
            print("[WARNING] You are not allowed to spend money on OpenAI. Change the guard 'ALLOWED_TO_SPEND' to be able to spend money on OpenAI")

            if is_openai_key_enabled():
                safe_disable_openai_key()
        else:
            if not is_openai_key_enabled():
                safe_enable_openai_key()

        self._memory = memory
        self.__set_LLM(model_name, stream, callbacks)
    ####################################


    ####################################
    def ask(self, questions:list[str], dump:bool=False, log:bool=False, user:str="") -> list[str]:
        """You can ask a single question or a list of questions. In any case, the answer will be a list of responses. So if you ask a single question, the response will be a list with a single element. If you ask a list of questions, you may want to do answers[0]"""
        if type(questions) is not list:
            if type(questions) is str:
                questions = [questions]
            else:
                raise TypeError("Questions must be a list of strings")
        
        responses = []
        if dump: conversation:Conversation = Conversation()
        for user_input in questions:
            response = self._send_question_to_ai(user_input)
            response_str = response["result"]
            responses.append(response_str)
            if log:
                print("user:", user_input) 
                print("bot:", response_str, '\n')
            if dump: 
                if "source_documents" in response:
                    docs = [doc.dict() for doc in response["source_documents"]]
                else:
                    docs = []
                conversation.add_interaction(user_input, response_str, docs)
        if dump:
            self._dumping_conversation(conversation, user=user)

        if len(questions) != len(responses): raise Exception("Wtf. The number of questions and responses is different.")

        return responses
   
    def ask_with_metadata(self, questions:list[str], dump:bool=False, log:bool=False, user:str="") -> list[str]:
        """You can ask a single question or a list of questions. In any case, the answer will be a list of responses. So if you ask a single question, the response will be a list with a single element. If you ask a list of questions, you may want to do answers[0]"""
        if type(questions) is not list:
            if type(questions) is str:
                questions = [questions]
            else:
                raise TypeError("Questions must be a list of strings")
        
        responses = []
        if dump: conversation:Conversation = Conversation()
        for user_input in questions:
            response = self._send_question_to_ai(user_input)

            response_str = response["result"]
            are_metadata_present:bool = "source_documents" in response and len(response["source_documents"])>0 and isinstance(response["source_documents"][0], Document)
            if are_metadata_present: 
                docs = [doc.metadata for doc in response["source_documents"]]
            else:
                docs = []
            responses.append({"response":response_str, "metadata":docs})
            
            if log:
                print("user:", user_input) 
                print("bot:", response_str, '\n')
            if dump: 
                if are_metadata_present:
                    docs = [doc.dict() for doc in response["source_documents"]]
                else:
                    docs = []
                conversation.add_interaction(user_input, response_str, docs)
        if dump:
            self._dumping_conversation(conversation, user=user)

        if len(questions) != len(responses): raise Exception("Wtf. The number of questions and responses is different.")

        return responses


    def start_chatting_in_prompt(self, dump:bool=False):
        print("The bot is ready to chat! Type 'quit' or press <Ctrl+C> to exit.")

        if dump: conversation:Conversation = Conversation()
        while True:
            try:
                user_input = input(">")
                if user_input.strip() == "quit": raise KeyboardInterrupt
                
                response = self._send_question_to_ai(user_input)
                response_str = response["result"]
                print("bot:", response_str, '\n')
                if dump:
                    if "source_documents" in response:
                        docs :list[Document] = response["source_documents"]
                        docs = [doc.dict() for doc in docs]
                    else:
                       docs = []
                    conversation.add_interaction(user_input, response_str, docs)
            except(KeyboardInterrupt, SystemExit):
                print("Goodbye!")
                break
        if dump:
            self._dumping_conversation(conversation)
    ####################################


    ####################################
    def __set_LLM(self, model_name:str, stream:bool, callbacks:list=[]): # -> ChatOpenAI:
        self.__model_name = model_name
        
        if self.ALLOWED_TO_SPEND:
            self.__llm = LLMHelper.get_model(model_name=model_name,streaming=stream,callbacks=callbacks)
        else:
            print("[WARNING] You are not allowed to spend money on OpenAI. Change the guard 'ALLOWED_TO_SPEND' to be able to spend money on OpenAI")
            self.__llm = None
            
    def _get_model_name(self) -> str:
        return self.__model_name
    
    def _get_LLM(self): # -> ChatOpenAI:
        return self.__llm
    
    def _does_chat_have_history(self) -> bool:
        return self._memory
    ####################################

    ####################################
    @abstractmethod
    def _send_question_to_ai(self, question: str) -> str:
        raise NotImplementedError("This method should be implemented in a subclass")
    
    @abstractmethod
    def _dumping_conversation(self, conversation: Conversation, user:str="") -> None:
        raise NotImplementedError("This method should be implemented in a subclass")
    
    @abstractmethod
    def flash_history(self) -> None:
        raise NotImplementedError("This method should be implemented in a subclass")
    ####################################

#region Non-streaming, non-retriever/retriever Chatbots
class ChatbotRetriever(GenericChatbot):
    """Use it if you want to use a retriever to find the documents to pass to the LLM. 
    
    If you don't want to use a retriever, use ChatbotSimple instead.

    <code>
        model_name = ModelNameHelper.Text2Text.GPT3_5_TURBO()
        db = ... # Get the db from somewhere
        chain_type = ChainHelper.AllowedChainType.STUFF
        search_type = RetrieverHelper.AllowedSearchType.SIMILARITY
        search_kwargs = RetrieverHelper.Args2Kwargs.similarity()
        bot = ChatbotRetriever(
                    model_name=model_name, 
                    m_chain_type=chain_type, 
                    db=local_db, 
                    m_search_type=search_type, 
                    m_search_kwargs=search_kwargs
        )
        bot.ask("How to get started with MRTK3?")
    </code>
    """
    # Some useful links:
    #   tutorial:       https://python.langchain.com/docs/use_cases/question_answering/
    #   Retriever (return di as_retriever()):   https://api.python.langchain.com/en/latest/schema/langchain.schema.vectorstore.VectorStoreRetriever.html#langchain.schema.vectorstore.VectorStoreRetriever
    # Example of map-reduce chain prompts: https://www.reddit.com/r/LangChain/comments/12kr7cb/validation_error_for_mapreducedocumentschain/

    def __init__(self, model_name, m_chain_type, db, m_search_type, m_search_kwargs, memory:bool=False, extra_prompt_context:str=""):
        super().__init__(model_name, memory=memory)

        if db is None or db.as_retriever is None:
            raise ValueError("The database must be a valid VectorStore.")
        
        self.__set_prompt(extra_prompt_context)
        self.__set_retriever(db, m_search_type, m_search_kwargs)
        self.__set_chain(m_chain_type)
    ####################################   


    ####################################
    
    def _get_chain(self): # -> RetrievalQA:
        return self.__qa_chain
    
    def _get_retriever(self): # -> VectorStoreRetriever:
        return self.__retriever
    
    def _get_prompt(self) -> PromptTemplate:
        return self.__QA_CHAIN_PROMPT
    
    def _get_retriever_search_type(self) -> str:
        return self.__set_retriever_search_type
    
    def _get_retriever_search_kwargs(self) -> str:
        return self.__set_retriever_kwargs
    
    def _get_chain_type(self) -> str:
        return self.__chain_type
    ####################################

    ####################################
    def __set_prompt(self, extra_prompt_context:str) -> None:
        if self.ALLOWED_TO_SPEND:
            if extra_prompt_context != "":
                extra_prompt_context = extra_prompt_context.strip()+"\n"

            ##
            # self.__QA_CHAIN_PROMPT = ChatPromptTemplates.CHATBOT_RETRIEVER__PROMPT_TEMPLATE_LLAMA(
            self.__QA_CHAIN_PROMPT = ChatPromptTemplates.CHATBOT_RETRIEVER__PROMPT_TEMPLATE(
                has_memory=self._does_chat_have_history(),
                extra_prompt_context=extra_prompt_context)             
            ############

    def __set_retriever(self, db:VectorStore, m_search_type, m_search_kwargs) -> None:
        self.__set_retriever_search_type = sanitize_filename(str(m_search_type)) 
        # self.__set_retriever_kwargs = dictionary_to_filename(m_search_kwargs)
        self.__set_retriever_kwargs = RetrieverHelper.SearchKwargs_FromArgs.stringify(m_search_kwargs)
        if self.ALLOWED_TO_SPEND:
            if db is None:
                self.__retriever = None
            else:
                self.__retriever = db.as_retriever(search_type=m_search_type, search_kwargs=m_search_kwargs)

    def __set_chain(self, m_chain_type) -> None:
        self.__chain_type = m_chain_type

        if self.ALLOWED_TO_SPEND: 
            m_llm = self._get_LLM()
            m_prompt = self._get_prompt()
            m_retriever = self._get_retriever()
            
            self.__qa_chain = RetrievalQAChainHelper(m_llm=m_llm, m_retriever=m_retriever, m_prompt=m_prompt, want_memory=self._does_chat_have_history(), m_chain_type=m_chain_type)
    ####################################
    

    ####################################
    def _send_question_to_ai(self, question: str) -> str:
        if self.ALLOWED_TO_SPEND: 
            chain = self._get_chain()
            # INPUT BE LIKE {'query':<question>}
            # OUTPUT BE LIKE {'result':<response> , 'source_documents':<documents>}
            # chain.combine_documents_chain.memory
            result = chain.invoke(question)
            return result
        else:
            docs = [Document(page_content="This is a placeholder document. You can't spend money on OpenAI.", metadata={"source": "placeholder"}), Document(page_content="This is a SECOND placeholder document. You can't spend money on OpenAI.", metadata={"source": "blabla"} )]
            docs = [doc.dict() for doc in docs]
            return {"result":"[Warning] Change the guard 'ALLOWED_TO_SPEND' to be able to spend money on OpenAI",
                    "source_documents":docs}
        
    def _dumping_conversation(self, conversation: Conversation, user:str="") -> None:
        now = time.strftime("%d%b%Y-%Hh%Mm%Ss")  # 11Nov2023-09h56m09s
        path_blocks = [CONVERSATIONS_PATH, 
            sanitize_filename(user), 
            sanitize_filename(self._get_model_name()), 
            sanitize_filename(self._get_chain_type()),
            sanitize_filename(self._get_retriever_search_type()),
            sanitize_filename(self._get_retriever_search_kwargs()),
            f'{sanitize_filename(now)}']
        dest_folder= os.path.join(*path_blocks)
        conversation.dump_conversation(dest_folder)


    def flash_history(self) -> None:
        if self.ALLOWED_TO_SPEND: 
            chain = self._get_chain()
            chain.combine_documents_chain.clear()
        else:
            print("[WARNING] You are not allowed to spend money on OpenAI. Change the guard 'ALLOWED_TO_SPEND' to be able to spend money on OpenAI")
    ####################################

class ChatbotSimple(GenericChatbot):
    """Use it if you want to use a retriever to find the documents to pass to the LLM. 
    
    If you don't want to use a retriever, use ChatbotSimple instead.

    <code>
        model_name = ModelNameHelper.Text2Text.GPT3_5_TURBO()
        bot = ChatbotSimple(model_name=model_name)
        bot.ask("How to get started with MRTK3?")
    </code>
    """
    
    def __init__(self, model_name:str, extra_prompt_context:str="", memory:bool=False):
        super().__init__(model_name, memory=memory, stream=False, callbacks=[])
        
        self.__set_prompt(extra_prompt_context)
        self.__set_chain()
    ####################################   


    ####################################
    def _get_chain(self): # -> RetrievalQA:
        return self.__qa_chain
    
    def _get_prompt(self) -> PromptTemplate:
        return self.__QA_CHAIN_PROMPT
    ####################################


    def __set_prompt(self, extra_prompt_context:str) -> PromptTemplate:
        if self.ALLOWED_TO_SPEND:
            if extra_prompt_context != "":
                extra_prompt_context = extra_prompt_context.strip()+"\n"

            self.__QA_CHAIN_PROMPT = ChatPromptTemplates.CHATBOT_SIMPLE__PROMPT_TEMPLATE(
                has_memory=self._does_chat_have_history(),
                extra_prompt_context=extra_prompt_context
            )

    def __set_chain(self) -> None:

        if self.ALLOWED_TO_SPEND: 
            m_llm = self._get_LLM()
            m_prompt = self._get_prompt()

            self.__qa_chain = LLMChainHelper(m_prompt=m_prompt, m_llm=m_llm, want_memory=self._does_chat_have_history())

    def _send_question_to_ai(self, question: str) -> str:
        if self.ALLOWED_TO_SPEND: 
            chain = self._get_chain()
            # INPUT BE LIKE {'question':<question>}
            # OUTPUT BE LIKE {'question':<question>, 'text':<response>}
            # result = chain.invoke({"question": question})
            result = chain.invoke(question)
            # print("DEBUG PRINT. The response from the chain is: ", result)
            return {"result": result["text"]}
        else:
            docs = [Document(page_content="This is a placeholder document. You can't spend money on OpenAI.", metadata={"source": "placeholder"}), Document(page_content="This is a SECOND placeholder document. You can't spend money on OpenAI.", metadata={"source": "blabla"} )]
            docs = [doc.dict() for doc in docs]
            return {"result":"[Warning] Change the guard 'ALLOWED_TO_SPEND' to be able to spend money on OpenAI",
                    "source_documents":docs}
        
    
    def _dumping_conversation(self, conversation: Conversation, user:str="") -> None:
        now = time.strftime("%d%b%Y-%Hh%Mm%Ss")  # 11Nov2023-09h56m09s
        path_blocks = [CONVERSATIONS_PATH, 
            sanitize_filename(self._get_model_name()),
            sanitize_filename("SIMPLE_(NO_RETRIEVER)"),
            f'{sanitize_filename(now)}']
        dest_folder= os.path.join(*path_blocks)
        conversation.dump_conversation(dest_folder)
    
    def flash_history(self) -> None:
        if self.ALLOWED_TO_SPEND: 
            chain = self._get_chain()
            chain.memory.clear()
        else:
            print("[WARNING] You are not allowed to spend money on OpenAI. Change the guard 'ALLOWED_TO_SPEND' to be able to spend money on OpenAI")
#endregion

from src.my_prompts import FrameworkPrompts
from src.utils.pinecone_utils import get_vectorstore_from_Pinecone
class ExpertBot():

    def __init__(self, model_name:str, k_documents_retrieved:int, have_memory:bool, use_scene_context:bool, db:VectorStore=None):
        extra_prompt_context = FrameworkPrompts.GET_ROLE_PROMPT()
        self.use_scene_context = use_scene_context
        self.bot:GenericChatbot = None
        if type(k_documents_retrieved) != int or k_documents_retrieved <= 0:
            self.bot = ChatbotSimple(model_name=model_name, extra_prompt_context=extra_prompt_context, memory=have_memory)
            print("WARNING: k_documents_retrieved must be an integer greater than 0. Using ChatbotSimple instead.")
        else:
            # filter = {"q_id": {"$eq": <STRING>"}}
            # filter = {"q_id": {"$nin": ["<STRING>", "<STRING>"]}}]"}}
            # m_search_kwargs=RetrieverHelper.SearchKwargs_FromArgs.similarity(k=2, filter={"q_id": {"$nin": ["47", "48", "49"]}} )
            self.bot = ChatbotRetriever(
                model_name=model_name,

                m_chain_type=ChainArgsHelper.AllowedChainType.STUFF,
                db=db, 
                # m_search_type=RetrieverHelper.AllowedSearchType.SIMILARITY_SCORE_THRESHOLD, 
                # m_search_kwargs=RetrieverHelper.SearchKwargs_FromArgs.similarity_score_threshold(k=6, score_threshold=0.65), #K!=2 causes problems with Pinecone+get metadata.... :(
                m_search_type=RetrieverHelper.AllowedSearchType.SIMILARITY, 
                m_search_kwargs=RetrieverHelper.SearchKwargs_FromArgs.similarity(k=k_documents_retrieved),

                extra_prompt_context=extra_prompt_context,
                memory=have_memory
            )
        
    def ask(self, question:str, user:str, dump:bool, log:bool) -> str:
        answer:list[str] = self.bot.ask(question, dump=dump, log=log, user=user)
        return answer[0]

    def ask_with_metadata(self, question:str, user:str, dump:bool, log:bool) -> str:
        answer:list[str] = self.bot.ask_with_metadata(question, dump=dump, log=log, user=user)
        return answer[0]
    
    def start_chatting_in_prompt(self, dump:bool=False):
        self.bot.start_chatting_in_prompt(dump=dump)

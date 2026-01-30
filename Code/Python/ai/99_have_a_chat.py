"""You can have a chat with an LLM model via terminal using this script."""
from src.utils.os_utils import make_python_ai_the_working_directory

from src.utils.langchain.chat import ExpertBot, ChatbotSimple
from src.utils.langchain.langchain_utils import ModelNameHelper
if __name__ == "__main__":
    make_python_ai_the_working_directory()

    # LLM_MODEL_NAME = ModelNameHelper.Text2Text.LLAMA_2__7B_Context32K()
    LLM_MODEL_NAME = ModelNameHelper.Text2Text.LLAMA_3__8B_Context16K()
    # LLM_MODEL_NAME = ModelNameHelper.Text2Text.MISTRAL_7B_OPENORCA_Q4()

    K = 0
    # bot = ExpertBot(model_name=LLM_MODEL_NAME, k_documents_retrieved=K, have_memory=False, use_scene_context=False)
    bot = ChatbotSimple(model_name=LLM_MODEL_NAME, memory=False)
    print("===>\t\tREADY TO CHAT<===")
    bot.start_chatting_in_prompt()
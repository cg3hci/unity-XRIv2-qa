import os
import json

LANGCHAIN_VERSION='0.3.*'
# LANGCHAIN_VERSION='0.0.35*'
GPT4ALL_MODELS_PATH = "C:\\Users\\<YOUR_USER_NAME>\\AppData\\Local\\nomic.ai\\GPT4All\\"

def __get_value_from_json(json_file, key, sub_key):
   #try to open the file at ./, if not found, try to open at ./Python/ai
   if not os.path.exists(json_file):
       json_file = os.path.join("Python", "ai", json_file)
   try:
       with open(json_file) as f:
           data = json.load(f)
           return data[key][sub_key]
   except Exception as e:
       print("Error: ", e)

_openaikey = __get_value_from_json("secrets.json", "langchain", "key")

#region OPENAI
__ENV_OPENAI_STR = "OPENAI_API_KEY"

def  is_openai_key_enabled():
    return __ENV_OPENAI_STR in os.environ and os.environ[__ENV_OPENAI_STR] == _openaikey


def safe_enable_openai_key():
    if __ENV_OPENAI_STR not in os.environ:
        os.environ[__ENV_OPENAI_STR] = _openaikey
        print("OpenAI key enabled")
    else:
        print("[WARNING: Should this happen?] OPENAI KEY ALREADY ENABLED")

def safe_disable_openai_key():
    if __ENV_OPENAI_STR in os.environ:
        del os.environ[__ENV_OPENAI_STR]
        print("OpenAI key disabled")
    else:
        print("[WARNING: Should this happen?] OPENAI KEY ALREADY DISABLED")
#endregion

#region PINECONE
def get_pinecone_key():
    return __get_value_from_json("secrets.json", "pinecone", "key")
#endregion

#region TELEGRAM
def get_telegram_token():
    return __get_value_from_json("secrets.json", "telegram", "token")
def get_telegram_chat_id():
    return __get_value_from_json("secrets.json", "telegram", "chat_id")
#endregion

#region REDDIT_API
def get_reddit_client_id():
    return __get_value_from_json("secrets.json", "reddit_dev", "client_id")

def get_reddit_client_secret():
    return __get_value_from_json("secrets.json", "reddit_dev", "client_secret")
#endregion

#region HUGGINGFACE
__ENV_HUGGINGFACE_STR = "HUGGINGFACEHUB_API_TOKEN"
def __get_huggingface_token():
    return __get_value_from_json("secrets.json", "huggingface", "token")

def safe_enable_huggingface_token():
    if __ENV_HUGGINGFACE_STR not in os.environ:
        os.environ[__ENV_HUGGINGFACE_STR] = __get_huggingface_token()
        print("HuggingFace Token enabled")
    else:
        print("[WARNING: Should this happen?] HUGGINGFACE TOKEN ALREADY ENABLED")
#endregion
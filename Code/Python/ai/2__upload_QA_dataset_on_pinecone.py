"""
This scripts embeds the QA dataset and uploads it to Pinecone cloud.
"""
import pandas as pd

from src.utils.langchain.langchain_utils import ModelNameHelper
from src.utils.os_utils import make_python_ai_the_working_directory, DomainSpecificKnowledgeSettings
from src.utils.pinecone_utils import clear_index, upload_dataset_to_Pinecone_cloud

if __name__ == '__main__':
    make_python_ai_the_working_directory()
    
    # If you want to clear the index before uploading new data, set this to True, run, False, run again.
    if DO_CLEAR_INDEX := False:
        clear_index()
        exit()

    MODEL_NAME = ModelNameHelper.MultiModal2Text.GPT4_OMNI()
    filename = "output" # without .csv extension

    path = DomainSpecificKnowledgeSettings.get_to_gpt_dataset_path(f'{filename}.csv', model_name=MODEL_NAME)

    def str2list(x:str):
        return x.strip("[]").replace("'", "").split(", ") if x != '[]' else list()
    df = pd.read_csv(path, encoding='utf-8')

    upload_dataset_to_Pinecone_cloud(df)


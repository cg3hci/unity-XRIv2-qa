from typing import TypedDict
from pathvalidate import sanitize_filename
from src.loaders.supergeneric_loader import SuperGenericLoader
from src.utils.os_utils import DomainSpecificKnowledgeSettings, safe_open_w
import os
import json

from src.my_prompts import FrameworkPrompts

class SavingFormat(TypedDict):
    foldername: str
    filename_no_extension: str
    qa_list: list['GenericInjectPrivateQALoader.QAFormat']

"""
    LIST OF METHODS TO IMPLEMENT:
        - get_QAs_data(self)->list[SavingFormat]
        
"""
from abc import abstractmethod
class GenericInjectPrivateQALoader(SuperGenericLoader):
    ################## CALL THESE IN THE MAIN FUNCTION ##################
    def build_docs(self, override_gpt_dataset:bool=False):
        for qa_file in self.get_QAs_data():
            self.__analyse_urls(qa_file, override_gpt_dataset)

    def get_qa_dataset_paths(self) -> list[str]:
        """Returns the list of paths where the QA json file are saved. The format is ["res/to_gpt_dataset/<MODEL_USED>/<TOPIC>/"]

        <b>Example: </b> ["res/to_gpt_dataset/gpt3-5-turbo/mrtk3/", ...]

        Returns:
            list[str]: _description_
        """
        if not hasattr(self, "__qa_dataset_path") or self.__qa_dataset_path is None:
            self.__qa_dataset_path = [DomainSpecificKnowledgeSettings.get_to_gpt_dataset_path(folder["foldername"], model_name="") for folder in self.get_QAs_data()]

        return self.__qa_dataset_path
    ####################################################################


    ################## METHODS TO IMPLEMENT ##################
    @abstractmethod
    def get_QAs_data(self)->list[SavingFormat]:
        """Get an array of dictionaries. Each dictionary should have the following keys: 'foldername', 'urls' """
        pass

    class QAFormat(TypedDict):
        question: str
        answer: str
    #####################################################################


    ################## INTERNAL STUFF ##################
    def __analyse_urls(self, qa_file:SavingFormat, override_gpt_dataset:bool=False):
        qa_list, folder, title = qa_file["qa_list"], qa_file["foldername"], qa_file["filename_no_extension"]

        print("Starting to process documents destined to folder: " + folder)

        ### Add all of the QAs
        qa_dataset_path = DomainSpecificKnowledgeSettings.get_to_gpt_dataset_path(folder, model_name="")

        ########## Generate the QA file and save it as a json file ##########
        try:
            qa_curr_path  =f"{qa_dataset_path}/{sanitize_filename(title)}-created via code.json"
            do_qa_curr_path_exist = os.path.exists(qa_curr_path)
            
            if not do_qa_curr_path_exist or override_gpt_dataset:
                # Save the json file
                dic_to_write = {
                    "messages": [
                        {
                            "role": "system",
                            "content": FrameworkPrompts.GET_ROLE_PROMPT()
                        },
                    ]
                }
                for qa in qa_list:
                    question, answer = qa["question"] , qa["answer"]
                    dic_to_write["messages"].append({
                            "role": "user",
                            "content": f"{question}"
                    })
                    dic_to_write["messages"].append({
                            "role": "assistant",
                            "content": f"{answer}"
                    })


                # Save the json in a file
                with safe_open_w(qa_curr_path) as f:
                    json.dump(dic_to_write, f)
        except Exception as e:
            print(f"Error while trying to generate and save the custom QA document in folder {qa_dataset_path}.\nError details:{e}\n")
            return
        ###
        print("Finished processing documents destined to folder: " + folder)
        #############################################################
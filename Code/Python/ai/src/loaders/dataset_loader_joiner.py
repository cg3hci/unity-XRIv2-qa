from src.loaders.supergeneric_loader import SuperGenericLoader
import glob
import json
import csv
from src.utils.os_utils import get_ancestor_path, DomainSpecificKnowledgeSettings

class LoaderJoinerDataset():

    def build_qa_dataset(self, loaders : list[SuperGenericLoader], model_name:str):
        folder_path_list = [f"{p}/*.json" for l in loaders for p in l.get_qa_dataset_paths()]

        output_filename = "output"
        jsonl_output_file = DomainSpecificKnowledgeSettings.get_to_gpt_dataset_path(f"{output_filename}.jsonl", model_name)
        csv_output_file =   DomainSpecificKnowledgeSettings.get_to_gpt_dataset_path(f"{output_filename}.csv", model_name)

        self.__create_dataset_as_JSONL_file(folder_path_list,jsonl_output_file)
        print("Created jsonl file")

        self.__create_dataset_as_CSV_file(jsonl_output_file, csv_output_file)
        print("Created csv file")


    def __create_dataset_as_JSONL_file(self, folder_path_list:list[str], output_file:str):
        paths:list[str] = []
        for folder_path in folder_path_list:
            paths.extend( glob.glob(folder_path, recursive=True) )
        
        with open(output_file, 'w+', encoding="utf8") as jsonl_file:
            # Iterate through each file in the folder
            for path in paths:

                # Check if the file is a JSON file
                if path.endswith('.json'):
                    with open(path, 'r') as json_file:
                        json_content = json.load(json_file)                        
                        jsonl_file.write(json.dumps(json_content) + '\n')

    def __create_dataset_as_CSV_file(self, jsonl_input_file:str, csv_output_file:str):
        with open(jsonl_input_file, 'r') as f:
            lines = f.readlines()

        data = []
        id=1
        for line in lines:
            json_data = json.loads(line)
            messages = json_data['messages']
            source_type = json_data['metadata']['source_type']
            source_origin = json_data['metadata']['source_origin']
            system_content = messages[0]['content']
            for i in range(1, len(messages), 2):
                user_question = messages[i]['content']
                assistant_response = messages[i + 1]['content'] if i + 1 < len(messages) else ''
                tags = messages[i]['tags'] if 'tags' in messages[i] else ''
                data.append([id, system_content, user_question, assistant_response, tags, source_type, source_origin])
                id += 1
                
        with open(csv_output_file, 'w+', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['id_num','system', 'question', 'answer', 'tags', 'source_type', 'source_origin'])
            csv_writer.writerows(data)



        
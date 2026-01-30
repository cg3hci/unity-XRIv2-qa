from abc import ABC, abstractmethod

class SuperGenericLoader(ABC):
    
    @abstractmethod
    def get_qa_dataset_paths(self)->list[str]:
        """Returns the list of paths where the QA json file are saved. The format is ["res/to_gpt_dataset/<MODEL_USED>/<TOPIC>/"]

        <b>Example: </b> ["res/to_gpt_dataset/<modelName>/<folderName>/", ...]

        Returns:
            list[str]: _description_
        """
        pass
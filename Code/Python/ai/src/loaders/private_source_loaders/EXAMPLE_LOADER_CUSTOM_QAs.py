from src.loaders.private_source_loaders.generic_private_loader import SavingFormat,  GenericInjectPrivateQALoader


class PRIVATE_QA_JSON_Loader(GenericInjectPrivateQALoader):
        
    def get_QAs_data(self)->list[SavingFormat]:
        custom_QAs:list[GenericInjectPrivateQALoader.QAFormat] = [
            # Topic #1
            {"question":"...?", "answer":"..."},
            # ...
            # Topic #N
            {"question": "...?", "answer": "..."},
        ]
        return [{
                "foldername":"<FOLDERNAME1>/.../<FOLDERNAME_N>",
                "filename_no_extension":"<FILENAME>",
                "qa_list":custom_QAs
                }]
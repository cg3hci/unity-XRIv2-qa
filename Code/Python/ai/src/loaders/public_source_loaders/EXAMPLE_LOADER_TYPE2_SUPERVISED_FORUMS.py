from bs4 import BeautifulSoup

from src.loaders.public_source_loaders.generic_public_loader import GenericLoaderType2_Forums_MarkedAnswer, InvalidDocumentExpection, SavingFormat
from src.utils.soup_utils import soup_to_markdown

# You can import links from another file (e.g.,
# from res.links.<YOUR_FOLDER_NAME>.<YOUR_.PY_FILE_NAME> import <YOUR_LIST_VARIABLE_NAME> as <NAME_YOU_LIKE>
NAME_YOU_LIKE = ["https://example.com/question1", "https://example.com/question2"]  # Example links



class supervisedForumExampleLoader(GenericLoaderType2_Forums_MarkedAnswer):
    def _needs_browserless_mode(self)->bool:
        # Return if you want to use Selenium browserless mode to load the webpage or not
        return False

    def _get_saving_foldernames(self)->list[SavingFormat]:
        """Get an array of dictionaries. Each dictionary should have the following keys: 'foldername', 'urls' """
        return [
            {
                "foldername": "<FOLDERNAME1>/.../<FOLDERNAME_N>", # folder structure to use inside res/orig/* and res/to_gpt_dataset/<MODEL_USED>/*
                "urls": NAME_YOU_LIKE,
                "source_origin": "The best forum"  # Metadata for vector store: source origin to add inside
            },
        ]
    
    ####################################################################################################
    def get_title_from_webpage(self, soup: BeautifulSoup, original_url:str) -> str:
        """Filename for the document, in our case, we search the header inside the webpage"""
        title = soup.find('QUERY CSS SELECTOR')
        return title.get_text().strip()

    def get_content_from_webpage(self, soup: BeautifulSoup, original_url:str) -> GenericLoaderType2_Forums_MarkedAnswer.WebpageContent:
        """You want to return a dictionary with the following keys: 'best_answer_body', 'question_body'
        If not found, raise InvalidDocumentExpection to skip the document"""
        # The body of the best answer
        best_answer_body = soup.select('QUERY CSS SELECTOR')

        # Skip the document if there is no a best answer (the one with the green checkmark)
        do_best_answer_exist = len(best_answer_body) != 0
        if not do_best_answer_exist: raise InvalidDocumentExpection("No best answer found")

        question_body = soup.select('<QUERY CSS SELECTOR>')
    
        return {
            'best_answer_body':soup_to_markdown(best_answer_body),
            'question_body':soup_to_markdown(question_body)
        }
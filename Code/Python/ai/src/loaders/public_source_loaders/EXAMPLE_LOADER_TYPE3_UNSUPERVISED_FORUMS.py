from bs4 import BeautifulSoup
from src.utils.soup_utils import soup_to_markdown
from src.loaders.public_source_loaders.generic_public_loader import GenericLoaderType3_Forums_NO_MarkedAnswer, InvalidDocumentExpection, SavingFormat

# You can import links from another file (e.g.,
# from res.links.<YOUR_FOLDER_NAME>.<YOUR_.PY_FILE_NAME> import <YOUR_LIST_VARIABLE_NAME> as <NAME_YOU_LIKE>
NAME_YOU_LIKE = ["https://example.com/question1", "https://example.com/question2"]  # Example links


class UNsupervisedForumExampleLoader(GenericLoaderType3_Forums_NO_MarkedAnswer):
    def _needs_browserless_mode(self)->bool:
        return False
    

    def _get_saving_foldernames(self)->list[SavingFormat]:
        return [{
            "foldername": "<FOLDERNAME1>/.../<FOLDERNAME_N>", # folder structure to use inside res/orig/* and res/to_gpt_dataset/<MODEL_USED>/*
            "urls":NAME_YOU_LIKE,
            "source_origin":"The best (unsupervised) forum"  # Metadata for vector store: source origin to add inside
        }]
        
    ####################################################################################################
    def get_title_from_webpage(self, soup: BeautifulSoup, original_url:str) -> str:
        title = soup.find('QUERY CSS SELECTOR').get_text().strip()
        return title
        


    def get_content_from_webpage(self, soup: BeautifulSoup, original_url:str) -> GenericLoaderType3_Forums_NO_MarkedAnswer.WebpageContent:
        try:
            comments = soup.select('QUERY CSS SELECTOR')
            are_there_comments = len(comments) != 0
            if not are_there_comments:
                raise InvalidDocumentExpection("No comments found")
            
            question_body = soup.select('QUERY CSS SELECTOR')
            question_body_str = soup_to_markdown(question_body)

            o:GenericLoaderType3_Forums_NO_MarkedAnswer.WebpageContent = {'question_body': question_body_str, 'all_answers': [], 'max_like':None}
            comments_data = []
            for idx, comment in enumerate(comments, start=1):
                b = comment.select_one('QUERY CSS SELECTOR')
                body_str = soup_to_markdown(b)
                comment_data:GenericLoaderType3_Forums_NO_MarkedAnswer.CommentInfo = {
                    "id": idx,
                    "body": body_str,
                    "likes": None,
                    "depth": 0
                }
                comments_data.append(comment_data)
            o['all_answers'] = comments_data
            return o

        except Exception as e:
            print(f"An error occurred: {e}")
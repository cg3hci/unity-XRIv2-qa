from bs4 import BeautifulSoup

from src.loaders.public_source_loaders.generic_public_loader import GenericLoaderType1_DocumentationLike, SavingFormat

from src.utils.soup_utils import soup_to_markdown

# You can import links from another file (e.g.,
# from res.links.<YOUR_FOLDER_NAME>.<YOUR_.PY_FILE_NAME> import <YOUR_LIST_VARIABLE_NAME> as <NAME_YOU_LIKE>
NAME_YOU_LIKE = ["https://example.com/blog1", "https://example.com/blog2"]  # Example links

class documentExampleLoader(GenericLoaderType1_DocumentationLike):
    def _needs_browserless_mode(self)->bool:
        # Return if you want to use Selenium browserless mode to load the webpage or not
        return False

    def _get_saving_foldernames(self)->list[SavingFormat]:
        """Get an array of dictionaries. Each dictionary should have the following keys: 'foldername', 'urls' """
        return [{
            "foldername":"<FOLDERNAME1>/.../<FOLDERNAME_N>", # folder structure to use inside res/orig/* and res/to_gpt_dataset/<MODEL_USED>/*
            "urls":NAME_YOU_LIKE,
            "source_origin": "The best blog" # Metadata for vector store: source origin to add inside
        }]

        
    def get_title_from_webpage(self, soup: BeautifulSoup, original_url:str) -> str:
        return soup.find('QUERY CSS SELECTOR').get_text().strip()
    

    def get_content_from_webpage(self, soup: BeautifulSoup, original_url:str) -> GenericLoaderType1_DocumentationLike.WebpageContent:
        article_body = soup.find("QUERY CSS SELECTOR")
        return {'all_text':soup_to_markdown(article_body)}
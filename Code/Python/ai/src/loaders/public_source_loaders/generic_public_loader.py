from typing import Any, TypedDict


import requests

from bs4 import BeautifulSoup, Tag

from pathvalidate import sanitize_filename
import re
import os

from src.my_prompts import FrameworkPrompts, LoaderPrompts

from src.loaders.supergeneric_loader import SuperGenericLoader
from src.utils import os_utils
from src.utils.langchain.chat import ChatbotSimple
from src.utils.os_utils import CHROME_DRIVER_PATH, DomainSpecificKnowledgeSettings

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

import json

import src.document_splitter as document_splitter

class SavingFormat(TypedDict):
    foldername: str
    urls: list[str]
    source_origin: str


class InvalidDocumentExpection(Exception):
    pass

"""
    LIST OF METHODS TO IMPLEMENT:
        - build_docs(self),
        - get_title_from_webpage(self, soup : BeautifulSoup) -> str,
        - get_content_from_webpage(self, soup : BeautifulSoup) -> str
        
"""
from abc import abstractmethod
class GenericScrapperLoader(SuperGenericLoader):
    class BuildDocsOption(TypedDict):
        skip_orig_generation:bool=False
        skip_qa_generation:bool=False
        override_orig_doc_if_exist:bool=False
        override_qa_doc_if_exist:bool=False

    ################## CALL THESE IN THE MAIN FUNCTION ##################
    # @abstractmethod
    def build_docs(self, opt:BuildDocsOption):
        # self.analyse_urls(<URLS>, <DEST_FOLDER>", split=False)
        for folder in self._get_saving_foldernames():
            # p = DomainSpecificKnowledgeSettings.get_orig_path(folder["foldername"])
            self._analyse_urls(folder["urls"], folder["foldername"], folder["source_origin"], opt)

    def get_qa_dataset_paths(self)->list[str]:
        if not hasattr(self, "__qa_dataset_path") or self.__qa_dataset_path is None:
             self.__qa_dataset_path = [DomainSpecificKnowledgeSettings.get_to_gpt_dataset_path(folder["foldername"], model_name=self.bot._get_model_name()) for folder in self._get_saving_foldernames()]

        return self.__qa_dataset_path
    ####################################################################


    ################## METHODS TO IMPLEMENT ##################
    @abstractmethod
    def _needs_browserless_mode(self)->bool:
        pass

    @abstractmethod
    def _get_saving_foldernames(self)->list[SavingFormat]:
        """Get an array of dictionaries. Each dictionary should have the following keys: 'foldername', 'urls' """
        pass

    @abstractmethod
    def get_title_from_webpage(self, soup:BeautifulSoup, original_url:str) -> str:
        pass

    @abstractmethod
    def get_content_from_webpage(self, soup:BeautifulSoup, original_url:str) ->Any:
        pass

    @abstractmethod
    def _content2MDcontent(self, url:str, title:str, content:Any)->str:
        pass

    class QAFormat(TypedDict):
        question: str
        answer: str
        source_type:str
    
    @abstractmethod
    def _content2QAcontent(self, url:str, title:str, content=Any)->QAFormat:
        pass
    #####################################################################


    ################## THESE MAY BE USELFUL ##################
    # If the WebPage have multiple types of structures, this is useful
    __DOCUMENT_TYPE=None
    def __reset_document_type(self):
        self.__DOCUMENT_TYPE=None
    
    # @abstractmethod
    def _eval_document_type_given_soap(self, soup:BeautifulSoup):
        return None

    # Useful if the page has multiple types of structures. Look at UnityLoader for an example
    def __set_document_type(self, document_type):
        self.__DOCUMENT_TYPE=document_type

    def _get_document_type(self):
        return self.__DOCUMENT_TYPE
    #####################################################################


    ################## INTERNAL STUFF ##################
    def __init__(self, model_name:str) -> None:
        self.bot = ChatbotSimple(model_name=model_name)

    def _analyse_urls(self, urls:list[str], folder:str, source_origin:str, opt:BuildDocsOption):
        print("Starting to process documents destined to folder: " + folder)
        # Iterate through every link in the Reference
        for i, link in enumerate(urls):
            if i >= 0:
                print(f"URL ({urls.index(link) + 1}/{len(urls)}) := {link}")

                # Add all of the documents created from the article
                self.__create_documents_from_webpage(link, folder, source_origin, opt)
                # time.sleep(random.uniform(1,3))
                # break
        print(f"Finished processing documents destined to folder: {folder}\n<--------------------------------------->\n")

    def __create_documents_from_webpage(self, url:str, folder:str, source_origin:str, opt:BuildDocsOption):
        orig_path = DomainSpecificKnowledgeSettings.get_orig_path(folder)
        qa_dataset_path = DomainSpecificKnowledgeSettings.get_to_gpt_dataset_path(folder, model_name=self.bot._get_model_name())

        self.__reset_document_type()

        # Wrap function in a try-except block to handle any potential exceptions
        try:
            soup = self.__scrape_webpage(url)
            soup = self.__clean_soup(soup)
            self.__set_document_type(self._eval_document_type_given_soap(soup))

            # Extract the title from the article
            title = self.get_title_from_webpage(soup, url)
            if not title:
                raise InvalidDocumentExpection("No title found")
                
            # Convert the webpage HTML to markdown
            try:
                # webpage_main_content = self.get_content_from_webpage(soup,url) 
                content:Any = self.get_content_from_webpage(soup,url)
            except InvalidDocumentExpection as e:
                print(f"Error processing the document content with {url} in folder {orig_path}\nError details:{e}\n")
                return

            print(f"Document scrapped correctly\n")
        except Exception as e:
            print(f"Error processing document with [{url}] in folder {orig_path}.\nError details:{e}\n")
            return

        ########## Save the orig file. It may be useful for debugging ##########
        if not opt["skip_orig_generation"]:
            try:
                orig_curr_path  =f"{orig_path}/{sanitize_filename(title)}.md"
                do_orig_curr_path_exist = os.path.exists(orig_curr_path)

                if not do_orig_curr_path_exist or opt["override_orig_doc_if_exist"]:
                    webpage_main_content = self._content2MDcontent(url=url, title=title, content=content)
                    webpage_main_content = self.__strip_links_from_markdown(webpage_main_content)
                    with os_utils.safe_open_w(orig_curr_path) as f:
                        f.write(webpage_main_content)
                    print(f"Document saved correctly in folder {orig_path}\n")
            except Exception as e:
                print(f"Error while trying to generate and save the MD document with [{url}] in folder {orig_path}.\nError details:{e}\n")
                return
        #############################################################

        ########## Generate the QA file and save it as a json file ##########
        if not opt["skip_qa_generation"]:
            try:
                qa_curr_path  =f"{qa_dataset_path}/{sanitize_filename(title)}-created via code.json"
                do_qa_curr_path_exist = os.path.exists(qa_curr_path)
                
                if not do_qa_curr_path_exist or opt["override_qa_doc_if_exist"]:
                    content:GenericScrapperLoader.QAFormat = self._content2QAcontent(url=url, title=title, content=content)
                    question, answer, source_type = content["question"], content["answer"], content["source_type"]

                    # Save the json file
                    dic_to_write = {
                        "metadata": {
                            "source_type":source_type,
                            "source_origin": source_origin
                        },
                        "messages": [
                            {
                                "role": "system",
                                "content": FrameworkPrompts.GET_ROLE_PROMPT()
                            },
                            {
                                "role": "user",
                                "content": f"{question}"
                            },
                            {
                                "role": "assistant",
                                "content": f"{answer}"
                            },
                        ]
                    }

                    # Save the json in a file
                    # with open(qa_curr_path, "w") as f:
                    with os_utils.safe_open_w(qa_curr_path) as f:
                        json.dump(dic_to_write, f, indent=4)
            except Exception as e:
                print(f"Error while trying to generate and save the QA document with [{url}] in folder {qa_dataset_path}.\nError details:{e}\n")
                return
        #############################################################


    def __scrape_webpage(self, url:str)->BeautifulSoup:
        if self._needs_browserless_mode():
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # Run in headless mode (no GUI)
            chrome_options.add_argument("--log-level=1")  # Disable Info level logs
            # chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"]) # Stronger disable logging

            # chrome_options.add_argument("--disable-logging") # Doesn't work
            
            # Initialize the Chrome WebDriver with the specified options
            service = Service(executable_path=CHROME_DRIVER_PATH)
            driver = webdriver.Chrome(service=service, options=chrome_options)
            # Navigate to the URL
            driver.get(url)
            # Get the page source (HTML content)
            html_content = driver.page_source
            # Close the WebDriver
            driver.quit()

            # Create a BeautifulSoup object for parsing
            soup = BeautifulSoup(html_content, 'html.parser')
        else:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            response = requests.get(url, headers=headers)       
            if response.status_code != 200:
                raise Exception(f"Failed to fetch page: {url}. Status code is {response.status_code}")
            soup = BeautifulSoup(response.content, 'html.parser')
            
        return soup

    def __clean_soup(self, soup:BeautifulSoup)->BeautifulSoup:
        # Remove imgs
        for img in soup.find_all('img'):
            img:Tag
            # print("AAA IMG: ", type(img))
            img.decompose()

        # Remove figures
        for figure in soup.find_all('figure'):
            figure:Tag
            # print("AAA FIGURE: ", type(figure))
            figure.decompose()

        # Remove empty <p> and <li>
        for p in soup.find_all(['p', 'li']):
            p:Tag
            if not p.text.strip():
                p.decompose()

        return soup

    def __strip_links_from_markdown(self, markdown_content:str)->str:
        # Make a RegEx pattern
        link_pattern = re.compile(r'\[(.*?)\]\((.*?)\)')

        # Find all matches in markdown_content
        matches = link_pattern.findall(markdown_content)

        for _match in matches:
            full_match = f'[{_match[0]}]({_match[1]})'

            # Replace the full link with just the text
            markdown_content = markdown_content.replace(full_match, _match[0])

        return markdown_content


class GenericLoaderType1_DocumentationLike(GenericScrapperLoader):

    def __init__(self, model_name:str):
        super().__init__(model_name=model_name)

    class WebpageContent(TypedDict):
        all_text: str

    @abstractmethod
    def get_content_from_webpage(self, soup:BeautifulSoup, original_url:str) ->WebpageContent:
        pass
    
    def _content2MDcontent(self, url:str, title:str, content:WebpageContent)->str:
        return content["all_text"]

    def _content2QAcontent(self, url:str, title:str, content=WebpageContent)->GenericScrapperLoader.QAFormat:
        prompt, output2qa = LoaderPrompts.GET_TYPE1_PROMPT(content["all_text"])
        answer:str = self.bot.ask(prompt)[0]
        question, answer = output2qa(content["all_text"], prompt, answer)
        return {"question": question, "answer": answer, "source_type":"WebArticle"}    
    
class GenericLoaderType2_Forums_MarkedAnswer(GenericScrapperLoader):
    
    def __init__(self, model_name:str):
        super().__init__(model_name=model_name)

    class WebpageContent(TypedDict):
        question_body: str
        best_answer_body: str


    @abstractmethod
    def get_content_from_webpage(self, soup:BeautifulSoup, original_url:str) ->WebpageContent:
        pass
    

    def _content2MDcontent(self, url:str, title:str, content:WebpageContent)->str:
        template = f"""# Question
{content['question_body']}

# Accepted Answer
{content['best_answer_body']}
"""
        return template

    def _content2QAcontent(self, url:str, title:str, content=WebpageContent)->GenericScrapperLoader.QAFormat:
        return {"question": content["question_body"], "answer": content['best_answer_body'], "source_type":"SupervisedForum"}

class GenericLoaderType3_Forums_NO_MarkedAnswer(GenericScrapperLoader):
    def __init__(self, model_name:str):
        super().__init__(model_name=model_name)
        self.THR_LIKE = 0.25

    class CommentInfo(TypedDict):
        id: int
        likes: int|None
        body: str
        depth: int

    class WebpageContent(TypedDict):
        question_body: str
        all_answers: list['GenericLoaderType3_Forums_NO_MarkedAnswer.CommentInfo']
        max_like: int|None

    @abstractmethod
    def get_content_from_webpage(self, soup:BeautifulSoup, original_url:str) ->WebpageContent:
        pass

    def __filter_comments(self, content:WebpageContent)->list[CommentInfo]:
        # IF max_like is None, it means therer is no a like-based system
        if content["max_like"] is None:
            filtered_comments = [commentInfo for commentInfo in content["all_answers"]]

        # IF max_like is an integer, then we have a like-based system
        elif type(content["max_like"]) == int: 
            # Define the Like threshold as X% w.r.t. max_like
            threshold = round(self.THR_LIKE * content["max_like"])

            # Filter comments based on the threshold
            filtered_comments = [commentInfo for commentInfo in content["all_answers"] if commentInfo["likes"] >= threshold]
        else:
            raise Exception("max_like must be an integer or None")
        
        return filtered_comments

    def _content2MDcontent(self, url:str, title:str, content=WebpageContent)->str:

        filtered_comments = self.__filter_comments(content)

        # Loop over filtered_comments and write a string containing all the comments in the format specified below.
        # Replay #{filtered_comments['id']} with  {filtered_comments['likes']} likes: {filtered_comments['body']}.
        # The number of '#' should be equal to the depth of the comment (filtered_comments['depth']).
        output=""
        for comment in filtered_comments:
            like_section = f" with {comment['likes']} likes" if comment['likes'] is not None else ""
            output += f"{'#' * (comment['depth']+1)} Reply {comment['id']}{like_section}:\n{comment['body']}\n\n"
    

        return output

    def _content2QAcontent(self, url:str, title:str, content:WebpageContent)->GenericScrapperLoader.QAFormat:
        output = self._content2MDcontent(url, title, content)

        filtered_comments = self.__filter_comments(content)

        # CONVERTING THE OUTPUT INTO A FINAL MARKDOWN STRING BY USING AN LLM
        template = LoaderPrompts.GET_TYPE3_PROMPT(content['question_body'], len(filtered_comments), output)
        answer:str = self.bot.ask(template)[0]

        # final_str = f"""#Question
        # {content['question_body']}

        # # Answer generated on Human-in-the-Loop feedback:
        # {answer}"""
        return {"answer": answer, "question": content['question_body'], "source_type":"UNSupervisedForum"}
    

import os
import fnmatch
import zipfile
import shutil

from colorama import Fore, Back, Style

import time

BASE_PATH = "./"

def dictionary_to_filename(d:dict)->str:
    if isinstance(d, str):
        return d
    elif not isinstance(d, dict):
        raise ValueError("The input should be a dictionary or a string")
    
    filename_parts = []

    for key, value in d.items():
        # Convert underscores in keys to double underscores
        key = key.replace('_', '__')
        
        # Convert key-value pairs to the desired format
        filename_parts.append(f'{key}_{value}')

    # Join the parts with double underscores
    filename = '__'.join(filename_parts)
    
    return filename

def get_ancestor_path(path:str, levels_up:int=1)->str:
    """Removes the last segment of a path.\n
    <b>Example: </b> Remove_last_segment("a/b/c") -> "a/b/"

    Args:
        path (str): A path

    Returns:
        str: That path without the last segment (up of one level in the hierarchy)
    """
    if not isinstance(levels_up, int) or levels_up<1:
        levels_up = 1
        
    for _ in range(levels_up):
        path_segments = path.split('/')
        last_segment = path_segments[-1]
        
        if not path.endswith('/'):
            if "." in last_segment:
                pass
            
            elif len(path_segments) > 1:
                print("[WARN] The input string should end with the separator. Exiting")
                return path
                
            else:
                print("[WARN] I've already reached the root. Exiting")
                return path
        
        if len(path_segments)==2 and last_segment=="":
            print("You have reached the root. Quitting")
            return path
            
        path = path.rstrip('/')  # Remove the trailing slash if it exists
        path = path[:path.rfind('/')+1] 

    return path

def safe_path_join(path:str, *paths:str)->str:
    """Make the join of paths and removes double slashes"""
    return os.path.normpath(os.path.join(path, *paths))
    
def make_python_ai_the_working_directory():
    try:
        script_directory = os.path.dirname("./Python/ai/")
        os.chdir(script_directory)
    except FileNotFoundError:
        pass

def safe_open_w(path, encoding='utf-8'):
    ''' Open "path" for writing, creating any parent directories as needed.'''
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    return open(path, 'w', encoding=encoding)

def clear_folder(path:str)->None:
    # delete all (includes the folder unfortunately)
    shutil.rmtree(path)
    # recreate the folder
    os.makedirs(path, exist_ok=True)

def zip_folder(folder_path, output_zip):
    """
    Zips the contents of a folder, including any nested folders and files.

    Parameters:
    - folder_path: str, the path to the folder to be zipped.
    - output_zip: str, the path to the output zip file.
    """
    try:
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    # Create the complete filepath of the file in the directory
                    file_path = os.path.join(root, file)
                    # Add file to zip
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)
    except FileNotFoundError:
        print('The file you are trying to zip does not exist.')
    except RuntimeError as e:
        print('An unexpected error occurred:', str(e))
    except zipfile.LargeZipFile:
        print('The file is too large to be compressed.')
# Example usage
# zip_folder('path/to/your/folder', 'output.zip')

################ Driver ################
# Needs Chrome version: 127.0.6533.99
CHROME_DRIVER_PATH = os.path.join(BASE_PATH, "chromedriver-YOUR-SYSTEM/YOUR_VERSION/chromedriver.exe")
#######################################


########### Recorded Conversations ###########
CONVERSATIONS_PATH = os.path.join(BASE_PATH,"conversations/")

def clear_conversations():
    clear_folder(CONVERSATIONS_PATH)

def zip_conversations()->str:
    """Save the conversations in a zip file with the current date and time as the name.
    Returns the name of the zip file.

    Returns:
        str: The name of the zip file
    """
    now = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    filename = f'conversations_{now}.zip'
    zip_folder(CONVERSATIONS_PATH, filename)
    return filename

##############################################


################ Documents ################
class DomainSpecificKnowledgeSettings:
    RES_PATH = "./res/"
    
    @classmethod
    def get_orig_path(cls, filename):
        orig_path = os.path.join(cls.RES_PATH, "orig/")
        return os.path.join(orig_path, filename)

    @classmethod
    def get_split_path(cls, filename):
        split_path = os.path.join(cls.RES_PATH, "split/")
        return os.path.join(split_path, filename)

    @classmethod
    def get_db_path(cls, filename):
        db_path = os.path.join(cls.RES_PATH, "db/")
        return os.path.join(db_path, filename)
    
    @classmethod
    def get_to_gpt_dataset_path(cls, filename:str, model_name:str)->str:
        """model_name = empty string is allowed"""

        if type(model_name) != type("str"):
            # model_name = ModelNameHelper.Text2Text.GPT3_5_TURBO()
            # print(Fore.WHITE + Back.YELLOW)
            # print(f"[ERROR in {cls.get_to_gpt_dataset_path.__name__}] model_name. Using default model name: '{model_name}'")
            # print(Style.RESET_ALL)
            raise ValueError("model_name is not a string")

        path = os.path.join(cls.RES_PATH, "to_gpt_dataset", model_name)
        return os.path.join(path, filename)
############################################


################ List Utils ################
def iterable_str_to_str(iterable: list[str]|set[str], prepend:str="", marker="-")->str:
    return prepend+(marker+(f'\n{marker}'.join(iterable)))

def str_to_iterable_str(s:str, prepend:str="", marker="-")->list[str]:
    # Remove the prepend
    if prepend:
        s = s[len(prepend):]
    # For each line, remove the marker
    lines = s.split("\n")
    lines = [line[len(marker):] for line in lines]
    return lines
##############################################


################ Stats about file system ################
def count_markdown_files(directory):
    markdown_files = fnmatch.filter(os.listdir(directory), "*.md")
    return len(markdown_files)

def count_words_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            words = content.split()
            return len(words)
    except FileNotFoundError:
        return 0

def count_words_in_directory(directory_path):
    total_words = 0
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            total_words += count_words_in_file(file_path)
    return total_words

def count_characters_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return len(content)
    except FileNotFoundError:
        return 0

def count_characters_in_directory(directory_path):
    total_characters = 0
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            total_characters += count_characters_in_file(file_path)
    return total_characters
##############################################


import time
def split_folder_and_filename_from_path(path:str) -> tuple[str, str]:
    folder = os.path.dirname(path)
    filename = os.path.basename(path)  
    return folder, filename
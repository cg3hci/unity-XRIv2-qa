from src.utils.os_utils import count_markdown_files, count_characters_in_directory, count_words_in_directory
from src.utils.os_utils import DomainSpecificKnowledgeSettings
import os
from math import ceil

def get_token_number_given_character_count(char_count):
    # 1 token = 4 character
    return char_count/4

def get_token_number_given_word_count(word_count):
    # 1 token = 3/4 word
    return word_count*(4/3)
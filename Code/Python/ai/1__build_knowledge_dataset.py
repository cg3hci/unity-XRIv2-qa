"""
This script allows to build the QA dataset from the public and private sources.
"""
from src.utils.os_utils import make_python_ai_the_working_directory

from src.utils.langchain.langchain_utils import ModelNameHelper

# Generic classes for type hinting
from src.loaders.supergeneric_loader import SuperGenericLoader 
from src.loaders.public_source_loaders.generic_public_loader import GenericScrapperLoader
from src.loaders.private_source_loaders.generic_private_loader import GenericInjectPrivateQALoader

from src.loaders.public_source_loaders.stackoverflow_loader import StackOverflowLoader
from src.loaders.public_source_loaders.stackexchange_loader import StackExchangeLoader
from src.loaders.public_source_loaders.unity_doc_loader import UnityLoader
from src.loaders.public_source_loaders.fmod_loader import FmodLoader
from src.loaders.public_source_loaders.discussions_unity_loader import DiscussionsUnityLoader
from src.loaders.public_source_loaders.reddit_loader import RedditLoader
from src.loaders.public_source_loaders.github_issue_loader import GithubIssueLoader
from src.loaders.public_source_loaders.github_discussions_loader import GithubDiscussionsLoader
from src.loaders.public_source_loaders.blogs_medium_loader import MediumLoader
from src.loaders.public_source_loaders.blogs_unity import LearnXRBlogLoader
from src.loaders.public_source_loaders.xri_doc_loader import XRILoader

# Specific private inject loaders
from src.loaders.private_source_loaders.mrtk3_custom_students_curiosity_loader import MRTK3_ExampleCustomQAs 

# Loader Joiner
from src.loaders.dataset_loader_joiner import LoaderJoinerDataset

if __name__ == '__main__':
    make_python_ai_the_working_directory()

    # Choose the model to be used for helping the generation (when needed) of the QA dataset.
    CHOSEN_MODEL = ModelNameHelper.MultiModal2Text.GPT4_OMNI()

    # Initialize the specific public loaders. 
    # Creating the instance does nothing by itself, but it allows you to call the build_docs() methods.
    list_of_scrapper_loaders : list[GenericScrapperLoader] = [
        # MRTK3Loader(CHOSEN_MODEL), # This is for MRTK3, not for XRI
        ### Type 1 -> Web Articles ###
        UnityLoader(CHOSEN_MODEL),
        MediumLoader(CHOSEN_MODEL),
        XRILoader(CHOSEN_MODEL),
        LearnXRBlogLoader(CHOSEN_MODEL),
        ### Type 2 -> Supervised Forums ###
        StackOverflowLoader(CHOSEN_MODEL),
        StackExchangeLoader(CHOSEN_MODEL), 
        FmodLoader(CHOSEN_MODEL),
        DiscussionsUnityLoader(CHOSEN_MODEL),
        GithubDiscussionsLoader(CHOSEN_MODEL),
        ### Type 3 -> Unsupervised Forums ###
        RedditLoader(CHOSEN_MODEL),
        GithubIssueLoader(CHOSEN_MODEL)
    ]

    # Initialize the specific private loaders.
    # Again, creating the instance does nothing by itself, but it allows you to call the build_docs() methods.
    list_of_custom_inject_loaders : list[GenericInjectPrivateQALoader] = [
        # MRTK3_ExampleCustomQAs()
    ]

    if DO_CREATE_ORIG_and_QnA := False:
        # Scrap and Create the knowledge documents.
        # For each loader, the build_docs() method will create the original (as .md) and QA (as .json) documents.
        opt = {"override_orig_doc_if_exist": not True,
                "override_qa_doc_if_exist": not True,
                "skip_orig_generation": False,
                "skip_qa_generation": False, 
        }
        for pub_loader in list_of_scrapper_loaders:
            pub_loader.build_docs(opt)

        for priv_loader in list_of_custom_inject_loaders:
            priv_loader.build_docs(opt)

    if DO_JOIN := True:
        # Join the knowledge documents. i.e. merge the .JSON files into a single .JSONL (json line) file.
        # The .JSONL file can be used (look at the other scripts) to fine-tune an OpenAI model and/or can be embedded and cached into a Pinecone (cloud) index.
        joiner = LoaderJoinerDataset()

        merged:list[SuperGenericLoader] = list_of_scrapper_loaders + list_of_custom_inject_loaders
        joiner.build_qa_dataset(merged, CHOSEN_MODEL)
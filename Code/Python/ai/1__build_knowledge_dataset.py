"""
This script allows to build the QA dataset from the public and private sources.
"""
from src.utils.os_utils import make_python_ai_the_working_directory

from src.utils.langchain.langchain_utils import ModelNameHelper

# Generic classes for type hinting
from src.loaders.supergeneric_loader import SuperGenericLoader 
from src.loaders.public_source_loaders.generic_public_loader import GenericScrapperLoader
from src.loaders.private_source_loaders.generic_private_loader import GenericInjectPrivateQALoader

from src.loaders.public_source_loaders.EXAMPLE_LOADER_TYPE1_ARTICLE import documentExampleLoader
from src.loaders.public_source_loaders.EXAMPLE_LOADER_TYPE2_SUPERVISED_FORUMS import supervisedForumExampleLoader
from src.loaders.public_source_loaders.EXAMPLE_LOADER_TYPE3_UNSUPERVISED_FORUMS import UNsupervisedForumExampleLoader

# Specific private inject loaders
from src.loaders.private_source_loaders.EXAMPLE_LOADER_CUSTOM_QAs import PRIVATE_QA_JSON_Loader as Private_QA_JSON_Loader

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
        documentExampleLoader(CHOSEN_MODEL),

        ### Type 2 -> Supervised Forums ###
        supervisedForumExampleLoader(CHOSEN_MODEL),
        ### Type 3 -> Unsupervised Forums ###
        UNsupervisedForumExampleLoader(CHOSEN_MODEL),
    ]

    # Initialize the specific private loaders.
    # Again, creating the instance does nothing by itself, but it allows you to call the build_docs() methods.
    list_of_custom_inject_loaders : list[GenericInjectPrivateQALoader] = [
        Private_QA_JSON_Loader()
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
            priv_loader.build_docs(opt["override_qa_doc_if_exist"])

    if DO_JOIN := True:
        # Join the knowledge documents. i.e. merge the .JSON files into a single .JSONL (json line) file.
        # The .JSONL file can be used (look at the other scripts) to fine-tune an OpenAI model and/or can be embedded and cached into a Pinecone (cloud) index.
        joiner = LoaderJoinerDataset()

        merged:list[SuperGenericLoader] = list_of_scrapper_loaders + list_of_custom_inject_loaders
        joiner.build_qa_dataset(merged, CHOSEN_MODEL)
import json
import time

from XRIv2_benchmark.benchmark_utils import Benchmark
from src.utils.langchain.chat import ExpertBot
from src.utils.langchain.langchain_utils import EmbeddingHelper, ModelNameHelper
from src.utils.os_utils import make_python_ai_the_working_directory
from src.utils.pinecone_utils import get_vectorstore_from_Pinecone


def Test(llm_model_name:str, ds,db, k_documents_retrieved:int, emb_model:EmbeddingHelper) -> dict:
    DO_SLEEP = False
    # print(f"===>\t\tRUNNING EXPERIMENT FOR MODEL {llm_model_name}<===")
            
    # Get the RAG Agent. If k=0, it will be a simple LLM
    bot = ExpertBot(model_name=llm_model_name, k_documents_retrieved=k_documents_retrieved, have_memory=False, use_scene_context=False, db=db)

    M = {
        "model_name":llm_model_name,
        "N_questions":len(ds),
        "K_documents_retrieved":k_documents_retrieved,
        "results":[]
    }

    # Generate the various scores for each question-answer pair
    for (qa_idx, qa) in enumerate(ds):
        print(f"\t\t===>[RAG (k={k_documents_retrieved})] Question {qa_idx + 1}/{len(ds)} <===")

        ######################## Generate Answers ####################################
        real_question, real_answer = qa["question"], qa["answer"]
        before = time.time()
        generated_answer = bot.ask_with_metadata(real_question, user="test", log=False, dump=False) #dump True)
        after = time.time()
        if DO_SLEEP: time.sleep(4)  # Sleep to avoid the API rate limit
        delta_time_generation = after - before

        ######################## Embeddings ##########################################
        real_answer_embedding = emb_model.embed_query(real_answer)
        gen_my_answer_embedding = emb_model.embed_query(generated_answer["response"])
        if DO_SLEEP: time.sleep(4)  # Sleep to avoid the API rate limit
        ##############################################################################

        # Append the results
        M["results"].append({
            "question": real_question,
            "real_answer": real_answer,
            "embeddings_real_answer": real_answer_embedding,
            "AI": {
                "generated_answer": generated_answer["response"],
                "embeddings": gen_my_answer_embedding,
                "time_generation": delta_time_generation,
                "metadata": [{"q_id":x["q_id"],"source_type": x["source_type"], "source_origin": x["source_origin"]} for x in generated_answer["metadata"]]
            },
            "metadata":{
                "difficulty": qa["difficulty"],
                "category": qa["category"]
            }
        })
    ##############################################################################
    return M

def Run_Experiment_Once(wd:str, ds, db, list_model_names:list[str], list_of_k_documents_retrieved:list[int]) -> None:
    # Dump the results to a JSON file
    def dump_json_results(wd:str, TO_DUMP:dict):
        now = time.strftime("%d%b%Y-%Hh%Mm%Ss")  # 11Nov2023-09h56m09s
        path = f"{wd}/results/RUN_numeric_test_results_{now}.json"
        with open(path, "w") as file:
            file.write(json.dumps(TO_DUMP, indent=4))
        print(f"Results saved to {path}")
    
    # Get an embedding model
    EMBED_MODEL = ModelNameHelper.Embedding.OLD_ADA()
    emb_model = EmbeddingHelper(EMBED_MODEL)

    TO_DUMP = []
    try:
        for llm_model_name in list_model_names:
            for k in list_of_k_documents_retrieved:
                print(f"\t===>RUNNING EXPERIMENT FOR MODEL {llm_model_name} WITH {k} DOCUMENTS RETRIEVED <===")
                M = Test(llm_model_name=llm_model_name, ds=ds, db=db, emb_model=emb_model, k_documents_retrieved=k)
                TO_DUMP.append(M)
            #### END K FOR K_DOCUMENTS ####
    except Exception as e:
        print(f"\n\nERROR: {e}. I'm going to dump the results so far.")
        dump_json_results(wd=wd, TO_DUMP=TO_DUMP)
        raise e


    dump_json_results(wd=wd, TO_DUMP=TO_DUMP)
############################################################################################################

if __name__ == "__main__":
    make_python_ai_the_working_directory()
    WORK_DIRECTORY = "XRIv2_benchmark/"

    # Load JSON data (for debugging purposes, using a dummy dictionary)
    if FAKE :=  False:
        filename = f"{WORK_DIRECTORY}TMP-benchmark.json"
    else:
        filename = f"{WORK_DIRECTORY}Real-benchmark.json"
    with open(filename, "r") as file:
        json_data = file.read()
    data_dict = json.loads(json_data)

    # Create the benchmark instance. If it's not valid, an exception will be raised.
    benchmark = Benchmark(data_dict)

    # Get the dataset we want (XRI dataset for Unity)
    platform = "Unity" 
    toolkit = "XRI" # "MRTK3"
    dataset = benchmark.get_dataset(platform, toolkit)

    # Get the vector store
    db = get_vectorstore_from_Pinecone()

    # Define the models to benchmark
    LIST_OF_MODEL_NAMES = [ModelNameHelper.Text2Text.GPT3_5_TURBO(), ModelNameHelper.MultiModal2Text.GPT4_OMNI_SUPER_CHEAP()]
    # LIST_OF_MODEL_NAMES = [ ModelNameHelper.Text2Text.LLAMA_2__7B_Context4K() ]
    # LIST_OF_MODEL_NAMES = [ ModelNameHelper.Text2Text.LLAMA_2__7B_Context32K() ]
    # LIST_OF_MODEL_NAMES = [ ModelNameHelper.Text2Text.LLAMA_3__8B_Context16K() ]
    # LIST_OF_MODEL_NAMES = [ ModelNameHelper.Text2Text.GEMMA_2_9B_it_q8() ]
    # LIST_OF_MODEL_NAMES = [ ModelNameHelper.Text2Text.Mixtral_2x7B_Q4() ]
    # LIST_OF_MODEL_NAMES = [ ModelNameHelper.Text2Text.Phi_3DOT5_Mini_3DOT8B_Q4() ]
    # LIST_OF_MODEL_NAMES = [ ModelNameHelper.Text2Text.Phi_3DOT5_Mini_3DOT8B_Q4() ]
    # LIST_OF_MODEL_NAMES = [ ModelNameHelper.Text2Text.MISTRAL_7B_OPENORCA_Q4() ]
    # LIST_OF_MODEL_NAMES = [ ModelNameHelper.Text2Text.DEEPSEEK_R1() ]
    #  
    N_RUNS:int = 3 # 3 if not deterministic, else 1
    MIN_DOC_RTVD, MAX_DOC_RTVD, STEP_DOC_RTV = 0, 12, 1 # K in 0..12, but you can change for your needs
    list_of_k_documents_retrieved = list(range(MIN_DOC_RTVD, MAX_DOC_RTVD + 1, STEP_DOC_RTV))
    

    for experiment_idx in range(N_RUNS):
        print(f"\n===> RUNNING EXPERIMENT {experiment_idx + 1}/{N_RUNS} <===")
        Run_Experiment_Once(wd=WORK_DIRECTORY, ds=dataset, db=db, list_model_names=LIST_OF_MODEL_NAMES, list_of_k_documents_retrieved=list_of_k_documents_retrieved,)
        print(f"===> END OF EXPERIMENT {experiment_idx + 1}/{N_RUNS} <===\n")
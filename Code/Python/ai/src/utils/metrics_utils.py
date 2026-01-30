from numpy import dot
from numpy.linalg import norm

def eval_cosine_similarity(emb1:list[float], emb2:list[float])->float:
    return dot(emb1, emb2)/(norm(emb1)*norm(emb2))


def eval_cosine_distance(emb1:list[float], emb2:list[float])->float:
    return 1 - eval_cosine_similarity(emb1, emb2)

# Load BLEU and ROUGE evaluators from Hugging Face's `evaluate` library
import evaluate
__bleu_evaluator = evaluate.load("bleu")
__meteor_evaluator = evaluate.load("meteor")
def eval_BLEU_score(a_real:str, a_gen:str)->float:
    """
    Calculate the BLEU score between a reference and a candidate answer.

    Args:
    - reference (str): The ground truth answer.
    - candidate (str): The generated answer by the LLM.

    Returns:
    - float: BLEU score.
    """
    # Use the BLEU evaluator from the `evaluate` library
    bleu_result = __bleu_evaluator.compute(predictions=[a_gen], references=[a_real])
    # print(bleu_result)
    return bleu_result['bleu']

def eval_METEOR_score(a_real:str, a_gen:str)->float:
    """
    Calculate the METEOR score between a reference and a candidate answer.

    Args:
    - reference (str): The ground truth answer.
    - candidate (str): The generated answer by the LLM.

    Returns:
    - float: METEOR score.
    """
    # Use the METEOR evaluator from the `evaluate` library
    meteor_result = __meteor_evaluator.compute(predictions=[a_gen], references=[a_real])
    return meteor_result['meteor']

from src.utils.langchain.langchain_utils import load_evaluator
__edit_distance_evaluator = load_evaluator("string_distance")

def eval_edit_distance(a_real:str, a_gen:str)->float:
    edit_distance = __edit_distance_evaluator.evaluate_strings(
        prediction=a_gen,
        reference=a_real,
    )
    return edit_distance['score']
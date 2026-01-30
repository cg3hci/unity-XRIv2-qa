import os
os.environ["OMP_NUM_THREADS"] = '1'

from settings import safe_enable_openai_key, safe_disable_openai_key, get_pinecone_key, LANGCHAIN_VERSION

from src.utils.langchain.langchain_utils import ModelNameHelper, EmbeddingHelper, VectorStore
from tqdm.auto import tqdm

import pandas as pd

from ast import literal_eval

# CONSTANTS:
EMBED_MODEL:'ModelNameHelper.Embedding.EmbeddingInfo' = ModelNameHelper.Embedding.OLD_ADA()

INDEX_NAME:str           = '<YOUR_INDEX_NAME>'
PINECONE_API_KEY:str     =  get_pinecone_key()
PINECONE_ENVIRONMENT:str = '<YOUR_ENV_NAME>'
NAMESPACE:str            = '<YOUR_NAMESPACE>'

if LANGCHAIN_VERSION == '0.0.35*':
    from langchain.vectorstores.pinecone import Pinecone as PineconeVectorStore
    import pinecone as pc
    pc.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT
    )

elif LANGCHAIN_VERSION == '0.3.*':
    from pinecone import  Pinecone
    from langchain_pinecone import PineconeVectorStore
    pc = Pinecone(api_key=PINECONE_API_KEY)

else:
    raise ValueError("Please select a valid version of Langchain")

def get_vectorstore_from_Pinecone(key_metadata = "answer") -> VectorStore:
    safe_enable_openai_key()
    embed = EmbeddingHelper(model_helper=EMBED_MODEL)
    index = pc.Index(INDEX_NAME)
    vectorstores = PineconeVectorStore(index, embed.get_embedding_model(), text_key=key_metadata, namespace=NAMESPACE)
    return vectorstores

def clamp_string_to_bytes(input_string: str, limit_bytes: int) -> str:
    # Encode the string to bytes using UTF-8
    byte_string = input_string.encode('utf-8')

    # If the byte length of the string is within the limit, return the original string
    if len(byte_string) <= limit_bytes:
        return input_string

    # If it's too long, truncate it byte by byte
    truncated_byte_string = byte_string[:limit_bytes]

    # Decode the truncated byte string back to a string, ignoring incomplete characters
    clamped_string = truncated_byte_string.decode('utf-8', errors='ignore')
    
    return clamped_string

def upload_dataset_to_Pinecone_cloud(df:pd.DataFrame):
    # safe_enable_openai_key()
    embed = EmbeddingHelper(model_helper=EMBED_MODEL)
    index = pc.Index(INDEX_NAME)
    BATCH_SIZE = 1 # Don't use too large batch size, otherwise it will crash

    # metadatas = []

    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        i_end = min(len(df), i+BATCH_SIZE)
        batch = df.iloc[i:i_end]

        to_upsert = [
            (
                str(row['id_num']),
                embed.embed_query(str(row['answer'])),
                # Total amount of metadata should be <= 34KB
                {
                    'q_id': str(row['id_num']),
                    'question': clamp_string_to_bytes(str(row['question']),5000),
                    # 'question_embed': str(embed.embed_query(str(row['question']))),
                    'answer': clamp_string_to_bytes(str(row['answer']), 30000),
                    'source_type':str(row['source_type']),
                    'source_origin':str(row['source_origin']),
                    # 'tags': row['tags'],
                }
            ) for _, row in batch.iterrows()
        ]

        # index.upsert(vectors=zip(ids, embeds, metadatas))
        index.upsert(vectors=to_upsert, namespace=NAMESPACE)
    ##############################

def clear_index():
    index = pc.Index(INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=NAMESPACE)
        print("Index cleared")
    except Exception as e:
        print("Error clearing index. Maybe the index is already empty?", e)
    cached_embeds_filename = 'embeds.csv'
    if os.path.exists(cached_embeds_filename):
        os.remove(cached_embeds_filename)
        print("Cached embeddings deleted")

class PineconeEmbeddingsUtilityClass:
    
    def __init__(self, vectors) -> None:
        self.answer_embeddings, self.question_embeddings, self.metadata, self.id = [], [], [], []
        for v in vectors:
            self.answer_embeddings.append( v["values"] )
            self.question_embeddings.append( v["valuesQ"])
            self.metadata.append( v["metadata"] )
            self.id.append( v["id"] )

    def __len__(self):
        assert len(self.answer_embeddings) == len(self.metadata) == len(self.id) == len(self.question_embeddings)
        return len(self.answer_embeddings)
    
    def to_pd_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "id": self.id,
            "metadata": self.metadata,
            "embeddings": self.answer_embeddings,
            "embeddingsQ": self.question_embeddings,
        })
    
    def __str__(self) -> str:
        for i in range(len(self)):
            print("id:", self.id[i], "metadata:", self.metadata[i], "embeddings:", self.answer_embeddings[i], "embeddingsQ:", self.question_embeddings[i])
    
def get_embeddings_from_Pinecone() -> PineconeEmbeddingsUtilityClass:
    """Returns a list of dictionaries. Each dictionary contains the following keys:
    - id: The id of the embedding
    - values: The vector of the embedding
    - metadata: The metadata of the embedding
    """
    filename = 'embeds.csv'
    # Check if the file already exists
    if os.path.exists(filename):
        print("Reading from file")
        df = pd.read_csv(filename, converters={'values': pd.eval, 'valuesQ':pd.eval, "metadata": literal_eval})
        vectores = df.to_dict('records')
    else:
        print("Reading from Pinecone")
        index = pc.Index(INDEX_NAME)

        stats = index.describe_index_stats()
        namespace_map = stats['namespaces']
        vectores = []
        for namespace in namespace_map:
            # vector_count = namespace_map[namespace]['vector_count']
            v = [0.0]*EMBED_MODEL["output_dim"]
            res = index.query(vector=[v], top_k=200, namespace=namespace, include_values=True, include_metadata=True)
            for match in res['matches']:
                embed = EmbeddingHelper(model_helper=EMBED_MODEL)
                D = match.to_dict()
                D['valuesQ'] = embed.embed_query(D['metadata']['question']) # Add the question embedding to the metadata
                vectores.append(D)
        
        # Save in a csv file
        df = pd.DataFrame(vectores)
        df.to_csv('embeds.csv', index=False)

    return PineconeEmbeddingsUtilityClass(vectores)
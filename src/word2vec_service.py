import os
import sys
import time
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../proto")))
import word2vec_pb2
import word2vec_pb2_grpc


def load_model():
    print("loading model")
    start_time = time.process_time()
    model = KeyedVectors.load_word2vec_format(
        os.environ["MODEL_PATH"], binary=True, limit=None
    )
    end_time = time.process_time()
    print(f"loaded model in {round(end_time - start_time, 1)} seconds")
    return model


class Word2Vec(word2vec_pb2_grpc.Word2VecServiceServicer):
    def __init__(self):
        self.model = load_model()
        self.vocabulary = set(self.model.key_to_index.keys())

    def EmbedWords(self, request, context):
        print("EmbedWords request received")
        words = request.words
        embeddings = [
            word2vec_pb2.Embedding(embedding=self.model.get_vector(word).tolist())
            for word in words
            if word in self.vocabulary
        ]
        return word2vec_pb2.EmbedWordsResponse(embeddings=embeddings)

    def Similarity(self, request, context):
        print("Similarity request received")
        embeddings1 = np.array([element.embedding for element in request.embeddings1])
        embeddings2 = np.array([element.embedding for element in request.embeddings2])
        avg_embedding1 = np.mean(embeddings1, axis=0)
        avg_embedding2 = np.mean(embeddings2, axis=0)
        return word2vec_pb2.SimilarityResponse(
            similarity=cosine_similarity([avg_embedding1], [avg_embedding2])[0][0]
        )

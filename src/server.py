import os
import sys
import grpc
from concurrent import futures
from word2vec_service import Word2Vec

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../proto")))
import word2vec_pb2_grpc


# start grpc server
def start():
    server = grpc.server(
        futures.ThreadPoolExecutor(), options=[("grpc.max_receive_message_length", -1)]
    )
    word2vec_pb2_grpc.add_Word2VecServiceServicer_to_server(Word2Vec(), server)
    address = f"{os.environ['DOMAIN']}:{os.environ['PORT']}"
    server.add_insecure_port(address)
    print(f"starting photos server on {address}")
    server.start()
    server.wait_for_termination()

import os

def create_if_necessary(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
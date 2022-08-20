import joblib
import pickle
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from azureml.core.model import Model
from rnnclass import PredicterRNN , MyCustomUnpickler
import json
       
def init():
    global model
    model_path = Model.get_model_path("lstmpicklemodel.pkl")
    ## A customed unpickler is defined to help refence the PredicterRnn class
    with open(model_path, 'rb') as f:
        unpickler = MyCustomUnpickler(f)
        model = unpickler.load()
        

def run(data):
    ## Set seed
    # obtain data as a pickle object
    # process raw data >> deserialize object
    try:
        data = pickle.load(open('./inference_data/vehicle1.pkl','rb'))
        # make prediction

        output_scores = model(data)
        _, idx = output_scores[-1].max(0)

        # you can return any data type as long as it is JSON-serializable
        return json.dumps({"prediction": int(idx)})
    except Exception as e:
        result = str(e)
        # return error message back to the client
        
        return json.dumps({"Error": result})

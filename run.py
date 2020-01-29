import warnings
from naive_classifier import naiveClassifier
from interpret import ecgInterpretation
from data_for_test import get_data_for_test
import pandas as pd
import numpy as np
import sys, os
from keras.models import load_model
from sacred import Experiment
from sacred.observers import MongoObserver

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

ex = Experiment("interpret")


@ex.config
def get_config():
    val_path = '/scratch/derickmath/datasets/base_dados_laudos_unificada_revista.csv'
    val_traces = '/scratch/derickmath/datasets/val_traces.csv'
    model_name = '/scratch/derickmath/deteccao_metricas/backup_model_best.hdf5'
#     val_path = '/srv/code/datasets/base_dados_laudos_unificada_revista.csv'
#     val_traces = '/srv/code/datasets/val_traces.csv'
#     model_name = '/srv/code/metricas/backup_model_best.hdf5'
    real = None
    noise = None
    sim = 100
    id_ecg = 1
    model = "naive"
    
@ex.capture
def one_execution(data, sim, id_ecg, model, model_name, real, noise):
    signals =  np.array([x for x in data['x'][:][id_ecg]])
    if(model == "naive"):
        classification_model = naiveClassifier()
        model_interp = ecgInterpretation()
        result = model_interp.execute(sim, classification_model, signals)
        return result
    elif(model == "tensorflow_resnet"):
        if(os.path.exists(model_name)):
            classification_model = load_model(model_name, compile=False)
            model_interp = ecgInterpretation()
            result = model_interp.execute(sim, classification_model, signals, T = True, realname = real, noisename = noise)
            return result
        else:
            raise Exception("no model")
            
@ex.capture
def execute(val_path, val_traces, sim, id_ecg, model, model_name, real, noise):
    # Get data
    data_ori = pd.read_csv(val_traces, sep = ";")
    exames = pd.read_csv(val_path, sep = ",")
    pd.options.display.max_columns = None
    data = get_data_for_test(path_to_traces=val_traces,
                                   path_to_val=val_path)
    
    if(id_ecg == 'all'):
        for i in range(len(data['x'])):
            print(">>> processing id", i, " <<<")
            one_execution(data, sim, id_ecg = i)
    else:
        one_execution(data, sim)
    

    
@ex.automain
def main(_run):

    warnings.filterwarnings("ignore")

    return execute()
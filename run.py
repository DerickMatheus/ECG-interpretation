import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from naive_classifier import naiveClassifier
from interpret import ecgInterpretation
from data_for_test import get_data_for_test
import pandas as pd
import numpy as np
import sys
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment("interpret")


@ex.config
def get_config():
    val_path = '/scratch/derickmath/datasets/base_dados_laudos_unificada_revista.csv'
    val_traces = '/scratch/derickmath/datasets/val_traces.csv'
    sim = 100
    id_ecg = 1
    
@ex.capture
def execute(val_path, val_traces, sim, id_ecg):
    # Get data
    data_ori = pd.read_csv(val_traces, sep = ";")
    exames = pd.read_csv(val_path, sep = ",")
    pd.options.display.max_columns = None
    data = get_data_for_test(path_to_traces=val_traces,
                                   path_to_val=val_path)
    signals =  np.array([x for x in data['x'][:][id_ecg]])

    model_interp = ecgInterpretation()
    model = naiveClassifier()
    return model_interp.execute(sim, model, signals)

    
@ex.automain
def main(_run):

    warnings.filterwarnings("ignore")

    return execute()
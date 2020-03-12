import warnings
from naive_classifier import naiveClassifier
from interpret import ecgInterpretation
import pandas as pd
import numpy as np
import h5py
import sys, os
from keras.models import load_model
from sacred import Experiment
from sacred.observers import MongoObserver

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

ex = Experiment("interpret")


@ex.config
def get_config():
    val_traces = 'data/ecg_tracings.hdf5'
    model  = 'data/model.hdf5'
    real   = None
    noise  = None
    sim    = 100
    id_ecg = 1
    output_name = None
    output_name_mean = None
    
@ex.capture
def one_execution(data, sim, id_ecg, model, real, noise, output_name, output_name_mean):
    signals = data[id_ecg]
    if(os.path.exists(model)):
        classification_model = load_model(model, compile=False)
        model_interp = ecgInterpretation(id_ecg)
        result = model_interp.execute(sim, classification_model, signals, T = True, realname = real, noisename = noise,
                                        output_name = output_name, output_name_mean = output_name_mean)
        return result
    else:
        raise Exception("no model")
            
@ex.capture
def execute(val_traces, sim, id_ecg, model, real, noise):
    # Get data
    with h5py.File(val_traces, 'r') as file:
        data = np.array(file['tracings'])
    pd.options.display.max_columns = None    
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

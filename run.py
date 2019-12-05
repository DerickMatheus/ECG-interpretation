import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from naive_classifier import naiveClassifier
from interpret import ecgInterpretation
from data_for_test import get_data_for_test
import pandas as pd
import numpy as np
import sys

val_path = '/scratch/derickmath/datasets/base_dados_laudos_unificada_revista.csv'
val_traces = '/scratch/derickmath/datasets/val_traces.csv'
# Get data
data_ori = pd.read_csv(val_traces, sep = ";")
exames = pd.read_csv(val_path, sep = ",")
pd.options.display.max_columns = None
data = get_data_for_test(path_to_traces=val_traces,
                               path_to_val=val_path)
signals =  np.array([x for x in data['x'][:][0]])

model_interp = ecgInterpretation()
model = naiveClassifier()
model.predict(signals)
print(model_interp.execute(int(sys.argv[1]), model, signals))

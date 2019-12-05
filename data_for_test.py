#%% Imports
import numpy as np
import pandas as pd
from data import diagnosis
from datetime import datetime
from ecg_datasets.read_xml import generate_signal

# For some reason the names are not the same in all the files
# This dictionary gives the correpondence between diagnosis names
# and the name on the test file
diagnosis_testfile = {"BAV1": "BAV1o", 
                      "BRD": "BRD",
                      "BRE": "BRE",
                      "BradSin": "Bradi",
                      "FibAtrial": "FA", 
                      "TaquiSin": "Taqui",
                      "FlutterAtrial": "Flutt"}


def prepare_atributes(df, date="2018-10-21"):
    N = len(df)
    # Get sex as a binary variable
    df.sexo = pd.Categorical(df.sexo)
    is_male = df.sexo.cat.codes.values
    
    def get_age(x):
        delta = datetime.strptime(date, '%Y-%m-%d') -  datetime.strptime(x, '%Y-%m-%d')
        delta_days = delta.days
        delta_year = delta_days/365.25
        age = np.floor(delta_year)
        return age
        
    df["Idade"] =  df.nascimento.map(get_age)
    # Get age into bins
    bins = pd.IntervalIndex.from_tuples([(-np.inf, 25), (25, 40), (40, 60), (60, 80), (80, np.inf)])
    # One hot encoding
    age_range = np.zeros((N, len(bins)))
    age_range[np.arange(N), pd.cut(df.Idade, bins).cat.codes.values] = 1
    # Attributes
    dict_attributes = df.to_dict('list')
    dict_attributes['is_male'] = is_male
    dict_attributes['age_range'] = age_range
    return dict_attributes


def get_data_for_test(path_to_val, path_to_traces, verbose=False):
    # Read csv
    df_val = pd.read_csv(path_to_val, sep=",", low_memory=False)
    df_traces = pd.read_csv(path_to_traces, sep=';', low_memory=False)

    # Get validation ids
    id_exam_val = df_val.id_exame.values
    n_entries_val, = id_exam_val.shape
    # Get ids
    id_exam_traces = df_traces.id_exame.values
    n_entries_traces, = id_exam_val.shape
    # Get intersection between the two of them
    id_exam = np.intersect1d(id_exam_val, id_exam_traces)
    n_entries, = id_exam.shape
    
    if verbose:
        print("First file:", n_entries_val, 
              "\nSecond file:", n_entries_traces, 
              "\nIntersection:", n_entries)
        
    # Get input signal
    def gen_signal(row):
        return generate_signal(row["amostra"], row["ganho"],
                               row["sensibilidade"], row["taxa_amostragem"])
    df_traces.amostra = df_traces.apply(gen_signal, axis=1)
    traces_all = df_traces.groupby("id_exame").amostra.apply(lambda x: np.stack(x.values).T).to_frame()
    traces_all.reset_index(drop=False, inplace=True)

    # Order dataframe according to some common index
    def redefine_index(dataframe, ids_cols, ids):
        # Remove duplicated rows
        dataframe.drop_duplicates(ids_cols, inplace=True)
        # Keep only entries present in both and change DataFrame order according to id_exam
        dataframe.set_index(ids_cols, drop=True, inplace=True)
        return dataframe.reindex(ids, copy=False)

    df_traces = redefine_index(df_traces, 'id_exame', id_exam)
    df_val = redefine_index(df_val, 'id_exame', id_exam)
    traces_all = redefine_index(traces_all, 'id_exame', id_exam)
    
    # Get diagnosis and attributes
    def check_diagnosis(string_input, current_name):
        """Return a list indicating if a diagnosis is present or not."""
        return [int(current_name in string_input[i]) for i in range(n_entries)]
    diag_string_input = df_val.classe.values
    for diag in diagnosis:
        df_traces[diag] = check_diagnosis(diag_string_input, diagnosis_testfile[diag])

    # Get input signal
    x = np.stack(list(traces_all.amostra.values))

    # Get diagnosis as an array
    y = df_traces.reindex(columns=diagnosis).values
    
    # Get attributes
    attributes = prepare_atributes(df_traces)

    # Put everything together as a dict
    data_dict = {'x': x, 'y': y, 'ids': df_traces.index, **attributes}
    return data_dict


if __name__ == "__main__":
    data = get_data_for_test(path_to_val='./datasets/testset/base_dados_laudos_unificada_revista.csv',
                             path_to_traces='./datasets/testset/val_traces.csv',
                             verbose=True)
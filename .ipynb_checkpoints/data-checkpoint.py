from keras.utils import HDF5Matrix
import numpy as np
import pandas as pd
import os.path
import sys

__all__ = ["get_data", "diagnosis", "train_val_split", "prepare_atributes"]

# ----- Data settings ----- #
diagnosis = ["BAV1", "BRD", "BRE", "BradSin", "FibAtrial", "TaquiSin", "FlutterAtrial"]
train_val_split = 0.98
# ------------------------- #


def verbose_mode(n_entries_before, n_entries_csv, n_entries_both, n_entries,
                 n_entries_unlimited, n_train, path_to_hdf5, path_to_csv):
    print("CLS: N={0} entries were found in '{1}'."
          .format(n_entries_unlimited, os.path.basename(path_to_hdf5)))
    if n_entries < n_entries_unlimited:
        print("      N={0} entries were ignored due to the maximum number of entries."
              .format(n_entries_unlimited - n_entries))
    print("CORE: N={0} entries were found in '{1}'.".format(n_entries_before, os.path.basename(path_to_csv)))
    if n_entries_csv < n_entries_before:
        print("      N={0} duplicated entries were removed during pre-processing."
              .format(n_entries_before - n_entries_csv))
    if n_entries_both < n_entries_csv:
        print("      N={0} entries were removed because their signals were not found in '{1}'."
              .format(n_entries_csv - n_entries_both, os.path.basename(path_to_hdf5)))
    if n_entries_both < n_entries:
        print("      We set default attributes to N={0} entries found only in '{1}'."
              .format(n_entries - n_entries_both, os.path.basename(path_to_hdf5)))
    print("Resulting in a total of {0} entries to be used for training and testing.".format(n_entries))
    print("N_train = {0} ({1:2d}%)/ N_test = {2} ({3:2d}%)"
          .format(n_train, int(np.round(100.0*n_train/n_entries)),
                  n_entries-n_train, 100-int(np.round(100.0*n_train/n_entries))))


def prepare_atributes(df):
    N = len(df)
    # Get sex as a binary variable
    df.Sexo = pd.Categorical(df.Sexo)
    is_male = df.Sexo.cat.codes.values
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


def get_data(path_to_hdf5, path_to_csv, max_entries=None, verbose=False):
    # Open signals dataset
    id_exam = HDF5Matrix(datapath=path_to_hdf5, dataset="id_exam")
    id_exam = np.array(id_exam.data)
    # Get number of entries
    n_entries_unlimited, = id_exam.shape
    if max_entries is not None:
        n_entries = min(n_entries_unlimited, max_entries)
        id_exam = id_exam[:n_entries]
    else:
        n_entries = n_entries_unlimited
    # Open CSV file
    df = pd.read_csv(path_to_csv, low_memory=False)
    n_entries_before, _ = df.shape
    # Remove duplicated rows
    df.drop_duplicates('N_exame', inplace=True)
    n_entries_csv, _ = df.shape
    # Entries not present in "id exam"
    mask = df['N_exame'].isin(id_exam)
    n_entries_both = sum(mask)
    # Keep only entries present in both and change DataFrame order according to id_exam
    df.set_index('N_exame', drop=True, inplace=True)
    df = df.reindex(id_exam, copy=False)
    # Get diagnosis
    df_diagnosis = df.reindex(columns=[d + "_F" for d in diagnosis])
    y = df_diagnosis.values
    # Get attributes
    attributes = prepare_atributes(df)
    # Get output array
    # Training and validation split
    n_train = int(train_val_split * n_entries)
    x_train = HDF5Matrix(datapath=path_to_hdf5, dataset="signal", end=n_train)
    x_val = HDF5Matrix(datapath=path_to_hdf5, dataset="signal", start=n_train, end=n_entries)
    y_train = y[:n_train, :]
    y_val = y[n_train:, :]

    # Verbose mode
    if verbose:
        verbose_mode(n_entries_before, n_entries_csv, n_entries_both, n_entries,
                     n_entries_unlimited, n_train, path_to_hdf5, path_to_csv)
    # Define data dictionary
    data_dict = {"x_train": x_train, "x_val": x_val, "y_train": y_train, "y_val": y_val}
    for key, value in attributes.items():
        data_dict[key+"_train"] = value[:n_train]
        data_dict[key+"_val"] = value[n_train:]
    return data_dict


if __name__ == "__main__":
    data = get_data(path_to_hdf5=sys.argv[1],
                    path_to_csv=sys.argv[2],
                    verbose=True)

import pandas as pd
import numpy as np
import neurokit as nk

from joblib import Parallel, delayed
import multiprocessing


class naiveClassifier():
    def __init__(self):
        self.ecg_process = [None] * 12
        self.signal = [None] * 12
        self.rrDuration = 0
        self.prDuration = 0
        self.pAmplitude = 0
        self.pduration = 0
        self.prDuration = 0
        self.qAmplitude = 0
        self.rAmplitude = 0
        self.sAmplitude = 0
        self.qrsDuration = 0
        self.qtDuration = 0
        self.stDuration = 0
        self.sttDuration= 0
        self.tDuration = 0
        self.tAmplitude = 0

    def detect_bav1o(self):
        return 0

    def detect_brd(self):
        return 0
    
    def detect_bre(self):
        return 0
    
    def detect_bradi(self):
        if(self.rrDuration == 0):
            self.compute_rrduration()
        if(self.rrDuration >= 400):
            return 1
        return 0
    
    def detect_fa(self):
        if(self.pDuration == 0):
            self.compute_rrduration()
        if(self.pDuration <= 30):
            return 1
        return 0
    
    def detect_taqui(self):
        if(self.rrDuration == 0):
            self.compute_rrduration()
        if(self.rrDuration <= 300):
            return 1
        return 0
    
    def detect_flutt(self):
        return 0

    def compute_rrduration(self):
        duration = []
        mean = []
        for i in range(12):
            for j in range(len(ecg_process[i]['ECG']['R_Peaks']) - 1):
                duration.append(ecg_process[i]['ECG']['R_Peaks'][j+1][0]
                                  - ecg_process[i]['ECG']['R_Peaks'][j][0])
            mean.append(np.mean(duration))
        self.rrDuration = np.mean(mean)

    def compute_qrsduration(self):
        duration = []
        mean = []
        for i in range(12):
            for j in range(len(ecg_process[i]['ECG']['R_Peaks']) - 1):
                duration.append(ecg_process[i]['ECG']['R_Paeaks'][j+1][0]
                                  - ecg_process[i]['ECG']['Q_Waves'][j][0])
            mean.append(np.mean(duration))
        self.qrsDuration = np.mean(mean)

    def compute_prduration(self):
        duration = []
        mean = []
        for i in range(12):
            for j in range(len(ecg_process[i]['ECG']['R_Peaks']) - 1):
                duration.append(ecg_process[i]['ECG']['R_Peaks'][j+1][0]
                                  - ecg_process[i]['ECG']['P_Waves'][j][0])
            mean.append(np.mean(duration))
        self.prDuration = np.mean(mean)
    
    def compute_pduration(self):
        duration = []
        mean = []
        for i in range(12):
            for j in range(len(ecg_process[i]['ECG']['P_Waves']) - 1):
                duration.append(ecg_process[i]['ECG']['P_Waves'][j+1][0]
                                  - ecg_process[i]['ECG']['P_Waves'][j][0])
            mean.append(np.mean(duration))
        self.pDuration = np.mean(mean)

    def read_data(self, data):
        if(type(data) != type([])):
            for i in range(12):
                self.signal[i] = np.array([x[i] for x in data[0]])
        else:
            self.signal = data
        meanrr = []
        for i in range(12):
            rrduration = []
            self.ecg_process[i] = nk.ecg_process(self.signal[i],
                                            sampling_rate = 400,
                                            hrv_features = None,
                                            filter_type = 'FIR')
            self.ecg_process[i]['ECG']['R_Peaks'] = [[y, x] for x, y in
                                                enumerate(self.ecg_process[i][
                                                    'ECG']['R_Peaks'])]


    def single_predict(self, data):
        self.read_data(data)
        result = np.zeros(7)
        if(self.detect_bav1o):
            result[0] = 1
        if(self.detect_brd):
            result[1] = 1
        if(self.detect_bre):
            result[2] = 1
        if(self.detect_bradi):
            result[3] = 1
        if(self.detect_fa):
            result[4] = 1
        if(self.detect_taqui):
            result[5] = 1
        if(self.detect_flutt):
            result[6] = 1
        return result

    def predict(self, data):
        num_cores = multiprocessing.cpu_count()
        if(len(np.shape(data)) == 2):
            result = self.single_predict(data)
        else:
            result = Parallel(n_jobs = num_cores)(
                delayed(self.single_predict)(x) for index,x in enumerate(data))
        return result

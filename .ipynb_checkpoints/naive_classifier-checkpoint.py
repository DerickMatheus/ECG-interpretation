import pandas as pd
import numpy as np
import neurokit as nk
#!/usr/bin/env python
# coding: utf-8
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib import pyplot as plt
import neurokit as nk
import random
import datetime

from joblib import Parallel, delayed
import multiprocessing

class naiveClassifier():
    def __init__(self):
        self.ecg_process = [None] * 12
        self.signal = [None] * 12
        self.vrate = 0
        self.arate = 0
        self.prDuration = 0
        self.pAmplitude = 0
        self.pDuration = 0
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
        if(self.prDuration == 0):
            self.compute_prduration()
        return np.sum(self.prDuration - 2000)

    def detect_brd(self):
        if(self.qrsDuration == 0):
            self.compute_qrsduration()
        return np.sum(0.12 - self.qrsDuration/400)
    
    def detect_bre(self):
        if(self.qrsDuration == 0):
            self.compute_qrsduration()
        return np.sum(0.12 - self.qrsDuration/400)
    
    def detect_bradi(self):
        if(self.rrDuration == 0):
            self.compute_rrduration()
        return np.sum(self.vrate - 400)
    
    def detect_fa(self):
        if(self.arate == 0):
            self.compute_arate()
        return np.abs(60 - np.sum(self.arate))
    
    def detect_taqui(self):
        if(self.vrate == 0):
            self.compute_vrate()
        return np.abs(100 - np.sum(self.vrate))
    
    def detect_flutt(self):
        return 0

    def compute_vrate(self):
        return len(self.ecg_process[6]['ECG']['R_Peaks']) * 6
    
    def compute_arate(self):
        return len(self.ecg_process[6]['ECG']['P_Waves']) * 6
    
    def compute_rrduration(self):
        duration = []
        mean = []
        for i in range(12):
            for j in range(len(self.ecg_process[i]['ECG']['R_Peaks']) - 1):
                duration.append(self.ecg_process[i]['ECG']['R_Peaks'][j+1][0]
                                  - self.ecg_process[i]['ECG']['R_Peaks'][j][0])
            mean.append(np.mean(duration))
        self.rrDuration = np.mean(mean)

    def compute_qrsduration(self):
        duration = []
        mean = []
        for i in range(12):
            lenS = len(self.ecg_process[i]['ECG']['S_Waves'])
            lenQ = len(self.ecg_process[i]['ECG']['Q_Waves'])
            k = 0
            j = 0
            while(k < lenQ and j < lenS - 1):
                if(self.ecg_process[i]['ECG']['S_Waves'][j+1][1] == self.ecg_process[i]['ECG']['Q_Waves'][k][1] + 1):
                    duration.append(self.ecg_process[i]['ECG']['S_Waves'][j+1][0]
                                      - self.ecg_process[i]['ECG']['Q_Waves'][k][0])
                    k += 1
                    j += 1
                else:
                    if(k < j):
                        k += 1
                    else:
                        j += 1
            if(len(duration) != 0):
                mean.append(np.mean(duration))
        mean = np.array(mean)[~np.isnan(mean)]
        if(len(mean) != 0):
            self.qrsDuration = np.mean(mean)
        else:
            self.qrsDuration = -1

    def compute_prduration(self):
        duration = []
        mean = []
        for i in range(12):
            lenR = len(self.ecg_process[i]['ECG']['R_Peaks'])
            lenP = len(self.ecg_process[i]['ECG']['P_Waves'])
            k = 0
            j = 0
            while(k < lenP and j < lenR - 1):
                if(self.ecg_process[i]['ECG']['R_Peaks'][j+1][1] == self.ecg_process[i]['ECG']['P_Waves'][k][1] + 1):
                    duration.append(self.ecg_process[i]['ECG']['R_Peaks'][j+1][0]
                                      - self.ecg_process[i]['ECG']['P_Waves'][k][0])
                    k += 1
                    j += 1
                else:
                    if(k < j):
                        k += 1
                    else:
                        j += 1
            mean.append(np.mean(duration))
        self.prDuration = np.mean(mean)
    
    def compute_pduration(self):
        duration = []
        mean = []
        for i in range(12):
            for j in range(len(self.ecg_process[i]['ECG']['P_Waves']) - 1):
                duration.append(self.ecg_process[i]['ECG']['P_Waves'][j+1][0]
                                  - self.ecg_process[i]['ECG']['P_Waves'][j][0])
            mean.append(np.mean(duration))
        self.pDuration = np.mean(mean)

    def read_data(self, data):
        if(type(data) != type([])):
                self.signal = np.transpose(data)
        else:
            self.signal = data
        meanrr = []
        for i in range(12):
            rrduration = []
            try:
                self.ecg_process[i] = nk.ecg_process(self.signal[i],
                                                sampling_rate = 400,
                                                hrv_features = None,
                                                filter_type='FIR')

                self.ecg_process[i]['ECG']['R_Peaks'] = [[y, x] for x, y in
                                                    enumerate(self.ecg_process[i][
                                                        'ECG']['R_Peaks'])]

            except:
                print("error on ecg segmentation\n derviv = ", i)
                raise NameError('ECG segmentation error')
        self.compute_pduration()
        self.compute_prduration()
        self.compute_qrsduration()
        self.compute_rrduration()
        self.compute_arate()
        self.compute_vrate()


    def single_predict(self, data):
        self.read_data(data)
        result = np.zeros(7)
        
        result[0] = self.detect_bav1o()
        result[1] = self.detect_brd()
        result[2] = self.detect_bre()
        result[3] = self.detect_bradi()
        result[4] = self.detect_fa()
        result[5] = self.detect_taqui()
        result[6] = self.detect_flutt()
        return result

    def predict(self, data):
        num_cores = multiprocessing.cpu_count()
        if(len(np.shape(data)) == 2):
            result = self.single_predict(data)
        else:
            result = Parallel(n_jobs = num_cores)(
                delayed(self.single_predict)(x[0]) for index,x in enumerate(data))
        return result

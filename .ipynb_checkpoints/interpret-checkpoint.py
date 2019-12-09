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

class ecgInterpretation():
    def __init__(self):
        self.result = []
        self.diagnosis_derivations = ["D1", "D2", "D3", "AVL", "AVF", "AVR",
                                      "V1", "V2", "V3", "V4", "V5", "V6"]
        self.diagnosis =["BAV1o", "BRD", "BRE", "Bradi", "FA", "Taqui",
                         "Flutt"]
        self.tests_wave = ['p', 't', 'q', 'r', 's', 'p_', 't_', 'q_', 'r_', 's_']
        self.tests_period = ['qrs', 'pr', 'st', 'stt', 'qt', 'rr']
        self.true_label = 0


    def get_amplitude(self, wave, deriv):
        if(wave == 'r' or wave == 'qrs'):
            if(deriv <= 6):
                amplitude = random.uniform(0.2, 0.25)
            else:
                amplitude = random.uniform(0.2, 0.35)
        elif(wave == 'q' or wave == 's'):
            if(deriv <= 6):
                amplitude = random.uniform(-0.2, -0.25)
            else:
                amplitude = random.uniform(-0.3, -0.35)
        elif(wave == 't'):
            if(deriv <= 6):
                amplitude = random.uniform(0.1, 0.15)
            else:
                amplitude = random.uniform(0.1, 0.15)
        else:
            amplitude = random.uniform(-0.2, 0.2)
        return(amplitude)

    def increase_amplitude(self, signal, interval_begin, interval_end, wave,
                           deriv, flag_plt = None):
        random.seed(datetime.datetime.now())
        signal_aux = deepcopy(signal)
        for i in range(min(len(interval_begin), len(interval_end))):
            amplitude = self.get_amplitude(wave, deriv)
            if(wave == 'p'):
                interval_begin[i][0] -= 20
            if(interval_begin[i][1] == interval_end[i][1]):
                mean_point = int(interval_begin[i][0] + (
                    interval_end[i][0] - interval_begin[i][0])/2)
                x = np.array([interval_begin[i][0], mean_point,
                              interval_end[i][0]])
                y = np.array([signal[interval_begin[i][0]],
                              signal[mean_point]+10*amplitude,
                              signal[interval_end[i][0]]])
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                xp = np.linspace(interval_begin[i][0], interval_end[i][0],
                                 interval_end[i][0] - interval_begin[i][0])
                k = 0
                for j in range(interval_begin[i][0], interval_end[i][0]):
                    signal[j] = p(xp[k])
                    k += 1
                if flag_plt is 1:
                    plt.plot(x/400, y/10, '.', xp/400, p(xp)/10, '-')
        if flag_plt is 1:
            n = len(signal)
            plt.plot(np.arange(n)/400, signal_aux/10)
            plt.ylim(-2,2)
            plt.show()

    def increase_duration(self, signal, interval_begin, interval_end):
        window = np.random.normal(5, 2, 1)
        for i in range(int(int(window[0])/2)):
            noise = np.random.normal(0, 2, 1)
            for j in interval_begin:
                signal[j - i] = signal[j] + noise
            noise = np.random.normal(0, 1, 1)
            for j in interval_end:
                signal[j + i] = signal[j + i] + noise

    def generate_noise(self, signal, peaks, type_of_noise, deriv):
        ecg = deepcopy(signal)
        
        if(type_of_noise == 'q'):
            self.increase_amplitude(ecg, np.array(peaks['ECG'][
                'Q_Waves_Onsets']),
                               np.array(peaks['ECG']['Q_Waves']),
                               type_of_noise, deriv)
        
        elif(type_of_noise == 'r'):
            self.increase_amplitude(ecg, np.array(peaks['ECG']['Q_Waves']),
                               np.array(peaks['ECG']['R_Peaks']),
                               type_of_noise, deriv)
                        
        elif(type_of_noise == 's'):
            self.increase_amplitude(ecg, np.array(peaks['ECG']['R_Peaks']),
                               np.array(peaks['ECG']['S_Waves']),
                               type_of_noise, deriv)
            
        elif(type_of_noise == 'qrs'):
            self.drift_qrs(ecg, np.array(peaks['ECG']['Q_Waves_Onsets']),
                      np.array(peaks['ECG']['Q_Waves_Onset']),
                      np.array(peaks['ECG']['S_Waves']))
            
        elif(type_of_noise == 'p'):
            self.increase_amplitude(ecg, np.array(peaks['ECG']['P_Waves']),
                               np.array(peaks['ECG']['Q_Waves_Onsets']),
                               type_of_noise, deriv)
        
        elif(type_of_noise == 't'):
            self.increase_amplitude(ecg, np.array(peaks['ECG'][
                'T_Waves_Onsets']),
                               np.array(peaks['ECG']['T_Waves_Ends']),
                               type_of_noise, deriv)

        elif(type_of_noise == 'q_'):
            self.increase_duration(ecg, np.array(peaks['ECG'][
                'Q_Waves_Onsets']),
                               np.array(peaks['ECG']['Q_Waves']))
        
        elif(type_of_noise == 'r_'):
            self.increase_duration(ecg, np.array(peaks['ECG']['Q_Waves']),
                               np.array(peaks['ECG']['R_Peaks']))
                        
        elif(type_of_noise == 's_'):
            self.increase_duration(ecg, np.array(peaks['ECG']['R_Peaks']),
                               np.array(peaks['ECG']['S_Waves']))
            
        elif(type_of_noise == 'qrs'):
            self.drift_qrs(ecg, np.array(peaks['ECG']['Q_Waves_Onsets']),
                      np.array(peaks['ECG']['Q_Waves_Onset']),
                      np.array(peaks['ECG']['S_Waves']))
            
        elif(type_of_noise == 'p_'):
            self.increase_duration(ecg, np.array(peaks['ECG']['P_Waves']),
                               np.array(peaks['ECG']['Q_Waves_Onsets']))
        
        elif(type_of_noise == 't_'):
            self.increase_duration(ecg, np.array(peaks['ECG']['T_Waves_Onsets']),
                               np.array(peaks['ECG']['T_Waves_Ends']))

        else:
            print("ERRO IN NOISE TYPE")
        return(ecg)

    def generate_noises_sim(self, sim, signal, peaks, noise_type, deriv, all_deriv):
        err = 0
        T = [None] * sim
        for j in range(sim):
            T[j] = []
            for i in range(12):
                prob = random.uniform(0, 1)
                if(i == deriv or (all_deriv and prob >= 0.5)):
                    T[j].append(self.generate_noise(signal[i], peaks[i],
                                               noise_type, i))
                else:
                    T[j].append(signal[i])
            T[j] = np.array([np.transpose(T[j])])
        return T

    def evaluate(self, scores, real_diagnose, sim, pathology):
        err = 0
        if(pathology == -1):
            err = []
        for j in range(sim):
            y_new_score = scores[j]
            if pathology == -1:
                err.append(y_new_score - real_diagnose)
            else:
                if isinstance(pathology, list):
                    change = 1
                    for f in pathology:
                        if(y_new_score[0][f] == real_diagnose[id][f]):
                            change = 0
                    if change == 1:
                        err += 1
                else:
                    if(y_new_score[pathology] != real_diagnose[pathology]):
                       err += 1
        if(pathology == -1):
            return(np.mean(err, axis = 0))
        return(1 - err/sim)

    def test_signal(self, sim, model, signal, peaks, noise_type,
                    real_diagnose, deriv, all_deriv, pathology = -1):
#        scores = [None] * sim
        T = self.generate_noises_sim(sim, signal, peaks, noise_type, deriv, all_deriv)
        if(all_deriv == False):
            print("simulation deriv = ", self.diagnosis_derivations[deriv])
        scores = model.predict(T)
#        scores = [model.predict(x) for x in T]
        err = self.evaluate(scores, real_diagnose, sim, pathology)
            
        return(err)

    def compute_score(self, sim, model, signal, nkprocess, all_deriv):
        start = time.time()
        nderivs = 12
        if(all_deriv == True):
            nderivs = 1
        
        for i in self.tests_wave:
            print("computing wave ", i)
            for j in range(nderivs):
                start_one_execution = time.time()
                res = self.test_signal(sim, model, signal, nkprocess, i,
                                  self.true_label, j, all_deriv)
                self.result.append([i, res])
        print(time.time() - start)


    def execute(self, sim, model, data, all_deriv = True):
        ## model most have a "predict" object

        ecg_process = [None] * 12
        signal = [None] * 12
        for i in range(12):
            signal[i] = np.array([x[i] for x in data])
            try:
                ecg_process[i] = nk.ecg_process(signal[i],
                                                sampling_rate = 400,
                                                hrv_features = None,
                                                filter_type='FIR')
                ecg_process[i]['ECG']['R_Peaks'] = [[y, x] for x, y in
                                                    enumerate(ecg_process[i][
                                                        'ECG']['R_Peaks'])]

            except:
                print("error on ecg segmentation\n derviv = ", i)
                raise NameError('ECG segmentation error')

        self.true_label = model.predict(signal)
        print(self.true_label)
        self.compute_score(sim, model, signal, ecg_process, all_deriv)
        return self.result
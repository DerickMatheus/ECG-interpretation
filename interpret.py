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
        self.derivations = ["D1", "D2", "D3", "AVL", "AVF", "AVR", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.diagnosis_derivations = ["D1", "D2", "D3", "AVL", "AVF", "AVR",
                                      "V1", "V2", "V3", "V4", "V5", "V6"]
        self.diagnosis =["BAV1o", "BRD", "BRE", "Bradi", "FA", "Taqui",
                         "Flutt"]
        self.tests_wave = ['qrs', 'p', 't', 'q', 'r', 's', 'AV_rate', 'pr', 'st', 'qt', 'axis']
        self.test = ['rhythm']
        self.tests_period = ['qrs', 'pr', 'st', 'stt', 'qt', 'rr']
        self.true_label = 0
        self.ecg_process = [None] * 12
        self.noiseplt = 'noise'
        self.realplt = 'real'
        
#     #teste manual segmentation
#     def manual_segment(self, peaks):
#         peaks['ECG']['Q_Waves'] = [[619,0],[1007,1],[1330,2],[1825,3],
#                                    [2221,4],[2625,5],[3061,6]]
#         peaks['ECG']['Q_Waves_Onsets'] = [[623,0],[987,1],[1321,2],[1790,3]
#                                           ,[2200,4],[2591,5],[3034,6]]
#         peaks['ECG']['P_Waves'] = [[600,0],[965,1],[1300,2],[1770,3],
#                                    [2180,4],[2570,5],[3000,6]]
#         peaks['ECG']['R_Peaks'] = [[633,0],[1020,1],[1343,2],[1837,3],
#                                    [2241,4],[2645,5],[3086,6]]
#         peaks['ECG']['S_Waves'] = [[653,0],[1047,1],[1370,2],[1854,3],
#                                    [2258,4],[2662,5],[3103,6]]
#         peaks['ECG']['T_Waves_Onsets'] = [[726,0],[1100,1],[1442,2],[1905,3],
#                                           [2338,4],[2731,5],[3179,6]]
#         peaks['ECG']['T_Waves_Ends'] = [[773,0],[1166,1],[1491,2],[1997,3],
#                                         [2396,4],[2790,5],[3240,6]]
#         return peaks
      
    #manual amplitudes change in every wave
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

    #change the amplitude of the ECG for a given wave and derivation
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
                              signal[mean_point]+amplitude,
                              signal[interval_end[i][0]]])
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                xp = np.linspace(interval_begin[i][0], interval_end[i][0],
                                 interval_end[i][0] - interval_begin[i][0])
                k = 0
                for j in range(interval_begin[i][0], interval_end[i][0]):
                    signal[j] = p(xp[k])
                    k += 1
                    
    #change the axis orietation of the ECG for a given derivation
    def change_axis(self, signal, deriv):
        init = self.ecg_process[deriv]['ECG']['Q_Waves']
        end  = self.ecg_process[deriv]['ECG']['S_Waves']
        i = 0
        j = 0
        while ((i < len(init)) and (j < len(end))):
                if(init[i][1] == end[j][1]):
                    for k in range(init[i][0], end[j][0]):
                        noise = np.random.normal(0, 0.05, 1)
                        signal[k] *= -1 
                        signal[k] += noise
                    i += 1
                    j += 1
                else:
                    if(i < j):
                        i += 1
                    else:
                        j += 1
        return signal
        
        
    #change the rhythm of the ECG
    def change_rhythm(self, signal):
        for deriv in [0, 1, 4]:
            p_waves = self.ecg_process[deriv]['ECG']['P_Waves']
            terminal_force = np.random.normal(0.22, 0.2, 1)[0]
            wave_duration = np.random.normal(40, 4, 1)[0]
            for waves in p_waves:
                central_point = waves[0]
                init = max(0, int(np.ceil(central_point - (wave_duration/2))))
                end = min(len(signal[deriv]) -1 , int(np.ceil(central_point + (wave_duration/2)) - 1))
                x = [init, central_point, end]
                y = [0, terminal_force, 0]
                z = np.polyfit(x, y, 3)
                f = np.poly1d(z)
                x_new = np.linspace(x[0], x[-1], end - init)
                y_new = f(x_new)
                signal[deriv][init:end] = y_new
        return signal
                    
    #A/V rante change
    def increase_rrduration(self, signal, deriv, n = None):
        
        if(n == None):
            n = random.randint(0, 20)

        for i in range(n):
            it = random.randint(0, len(self.ecg_process[deriv]['ECG']['R_Peaks']) - 2)
            rs = [self.ecg_process[deriv]['ECG']['R_Peaks'][it][0], self.ecg_process[deriv]['ECG']['R_Peaks'][it + 1][0]]
            signal = np.append(signal, signal[rs[0]:rs[1]])
            for i in range(rs[1] - rs[0]):
                pos = random.randint(0, len(signal) - 2)
                signal[pos] = (signal[pos] + signal[pos+1])/2
                signal = np.delete(signal, pos+1, axis = 0)
                
        return (signal)
    
    #random noise insertion
    def insert_noise(self, signal, interval_begin, interval_end, wave, deriv):   
        window = np.random.normal(5, 2, 1)
        i = 0
        j = 0
        while ((i < len(interval_begin)) and (j < len(interval_end))):
                if(interval_begin[i][1] == interval_end[j][1]):
                    for k in range(interval_begin[i][0], interval_end[j][0]):
                        noise = np.random.normal(0, 0.5, 1)
                        signal[k] += noise
                    i += 1
                    j += 1
                else:
                    if(i < j):
                        i += 1
                    else:
                        j += 1

    def generate_noise(self, signal, peaks, type_of_noise, deriv, n):
        ecg = deepcopy(signal)        

        if(type_of_noise == 'AV_rate'):
            ecg = self.increase_rrduration(ecg, deriv, n)
        
        elif(type_of_noise == 'q'):
            self.insert_noise(ecg, np.array(peaks['ECG'][
                'Q_Waves_Onsets']), np.array(peaks['ECG']['Q_Waves']),
                               type_of_noise, deriv)
        
        elif(type_of_noise == 'r'):
            self.insert_noise(ecg, np.array(peaks['ECG']['Q_Waves']),
                               np.array(peaks['ECG']['R_Peaks']),
                               type_of_noise, deriv)
                        
        elif(type_of_noise == 's'):
            self.insert_noise(ecg, np.array(peaks['ECG']['R_Peaks']),
                               np.array(peaks['ECG']['S_Waves']),
                               type_of_noise, deriv)
            
        elif(type_of_noise == 'p'):
            self.insert_noise(ecg, np.array(peaks['ECG']['P_Waves']),
                               np.array(peaks['ECG']['Q_Waves_Onsets']),
                               type_of_noise, deriv)
        
        elif(type_of_noise == 't'):
            self.insert_noise(ecg, np.array(peaks['ECG']['T_Waves_Onsets']),
                               np.array(peaks['ECG']['T_Waves_Ends']),
                               type_of_noise, deriv)
            
        elif(type_of_noise == 'qrs'):
            self.insert_noise(ecg, np.array(peaks['ECG']['Q_Waves']),
                               np.array(peaks['ECG']['S_Waves']),
                               type_of_noise, deriv)
            
        elif(type_of_noise == 'pr'):
            self.insert_noise(ecg, np.array(peaks['ECG']['P_Waves']),
                               np.array(peaks['ECG']['Q_Waves']),
                               type_of_noise, deriv)
            
        elif(type_of_noise == 'qt'):
            self.insert_noise(ecg, np.array(peaks['ECG']['Q_Waves']),
                               np.array(peaks['ECG']['T_Waves_Ends']),
                               type_of_noise, deriv)
            
        elif(type_of_noise == 'st'):
            self.insert_noise(ecg, np.array(peaks['ECG']['S_Waves']),
                               np.array(peaks['ECG']['T_Waves_Ends']),
                               type_of_noise, deriv)
            
        elif(type_of_noise == 'axis'):
            ecg = self.change_axis(ecg, deriv)
            
        elif(type_of_noise == 'rhythm'):
            ecg = self.change_rhythm(ecg)
            

        else:
            print("ERRO IN NOISE TYPE")
        return(ecg)
    
    def plot(self, ECG, name, peaks):
        x = np.arange(4096)


        fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(18,20))
        fig.tight_layout()
        i = 0
        j = 0
        k = 0
        p = np.array([x[0] for x in peaks[0]['ECG']['P_Waves']])
        p_ = np.array([x[0] for x in peaks[0]['ECG']['Q_Waves_Onsets']])
        q = np.array([x[0] for x in peaks[0]['ECG']['Q_Waves']])
        r = np.array([x[0] for x in peaks[0]['ECG']['R_Peaks']])
        s = np.array([x[0] for x in peaks[0]['ECG']['S_Waves']])
        t = np.array([x[0] for x in peaks[0]['ECG']['T_Waves_Onsets']])
        t_ = np.array([x[0] for x in peaks[0]['ECG']['T_Waves_Ends']])
        for y in ECG: 
            ax[j, i].plot(x/400, y/5)
            ax[j, i].plot(p/400, y[p]/5,  'rv', ms = 4)
            ax[j, i].plot(p/400, y[p_]/5,  'rv', ms = 4)
            ax[j, i].plot(q/400, y[q]/5,  'g^', ms = 4)
            ax[j, i].plot(r/400, y[r]/5,  'y<', ms = 4)
            ax[j, i].plot(s/400, y[s]/5,  'm>', ms = 4)
            ax[j, i].plot(t/400, y[t]/5,  'co', ms = 4)
            ax[j, i].plot(t_/400, y[t_]/5,  'co', ms = 4)

            #ax[j, i].legend(["sinal", "p", "p_", "q", "r", "s", "t", "t_"])
            
            major_ticksy = np.arange(-0.75, 0.75, 0.5)
            minor_ticksy = np.arange(-0.75, 0.75, 0.1)
            major_ticksx = np.arange(0, 10, 0.2)
            minor_ticksx = np.arange(0, 10, 0.05)
            ax[j, i].set_xlim(0, 10)
            ax[j, i].set_xlim(-0.75, 0.75)
            ax[j, i].grid(which='both')
            ax[j, i].set_xticks(major_ticksx)
            ax[j, i].set_xticks(minor_ticksx, minor=True)
            ax[j, i].set_yticks(major_ticksy)
            ax[j, i].set_yticks(minor_ticksy, minor=True)
            ax[j, i].grid(which='minor', alpha=0.2, color='r')
            ax[j, i].grid(which='major', alpha=0.5, color='r')
            ax[j, i].set_xticklabels( () )
            ax[j, i].set_yticklabels( () )
            ax[j, i].set_title(self.derivations[k])

            if(i == 5):
                ax[j, i].set_xlabel('t (seconds)')
            if(j == 0):
                ax[j, i].set_ylabel('x (mV)')
                
            
            
            j += 1
            k += 1
            if(j == 6):
                i += 1
                j = 0
        plt.savefig(name)
        plt.close()

    def generate_noises_sim(self, sim, signal, peaks, noise_type, deriv, all_deriv, n):
        err = 0
        T = [None] * sim
        self.plot(signal, self.realplt, peaks)
        for j in range(sim):
            T[j] = []
            if(noise_type == self.test[0]):
                T[j] = self.generate_noise(signal, peaks[0],
                                               noise_type, 0, n)
            else:
                for i in range(12):
                    prob = random.uniform(0, 1)
                    if(i == deriv or (all_deriv and prob >= 0.5)):
                        T[j].append(self.generate_noise(signal[i], peaks[i],
                                                   noise_type, i, n))
                    else:
                        T[j].append(signal[i])
            if(j == 0):
                self.plot(T[j], self.noiseplt + noise_type, peaks)
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
            for i in err:
                print(i)
            return(np.mean(err, axis = 0))
        return(1 - err/sim)

    def test_signal(self, sim, model, signal, peaks, noise_type,
                    real_diagnose, deriv, all_deriv, n, pathology = -1):
        T = self.generate_noises_sim(sim, signal, peaks, noise_type, deriv, all_deriv, n)
        if(all_deriv == False and noise_type != self.test):
            print("simulation deriv = ", self.diagnosis_derivations[deriv])
        T = np.squeeze(T, axis=1)
        scores = model.predict(T)
        #scores = [model.predict(x) for x in T]
        err = self.evaluate(scores, real_diagnose, sim, pathology)
            
        return(err)

    def compute_score(self, sim, model, signal, all_deriv, n):
        start = time.time()
        nderivs = 12
        if(all_deriv == True):
            nderivs = 1
        
        for i in self.tests_wave:
            print("computing wave ", i)
            for j in range(nderivs):
                start_one_execution = time.time()
                res = self.test_signal(sim, model, signal, self.ecg_process, i,
                                  self.true_label, j, all_deriv, n)
                print(i, res)
                self.result.append([i, res])
        print("computing ", self.test[0])
        res = self.test_signal(sim, model, signal, self.ecg_process, self.test[0],
                                  self.true_label, 0, all_deriv, n)
        print(self.test[0], res)
        self.result.append([self.test[0], res])
        print(time.time() - start)


    def execute(self, sim, model, data, all_deriv = True, T = False, realname = None, noisename = None):
        ## model most have a "predict" object

        if(realname != None or noisename != None):
            self.realplt = realname
            self.noiseplt = noisename
        
        signal = [None] * 12
        for i in range(12):
            signal[i] = np.array([x[i] for x in data])
#             try:
            self.ecg_process[i] = nk.ecg_process(signal[i],
                                            sampling_rate = 400,
                                            hrv_features = None,
                                            filter_type='FIR')
            self.ecg_process[i]['ECG']['R_Peaks'] = [[y, x] for x, y in
                                                enumerate(self.ecg_process[i][
                                                    'ECG']['R_Peaks'])]

            ######## remove this line
            #self.ecg_process[i] = self.manual_segment(self.ecg_process[i])
            ######## remove this line



        if(T == True):
            aux_signal = deepcopy(np.array([np.transpose(signal)]))
        else:
            aux_signal = signal
        self.true_label = model.predict(aux_signal)
        print(self.true_label)
        n = random.randint(0, 20)
        self.compute_score(sim, model, signal, all_deriv, n)
        return self.result
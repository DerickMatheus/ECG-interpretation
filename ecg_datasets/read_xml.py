import numpy as np
from xmljson import badgerfish as bf
from xml.etree.ElementTree import fromstring
from scipy.signal import (decimate, resample_poly)


def resample_ecg(trace, input_freq, output_freq=500):
    trace = np.atleast_1d(trace).astype(float)
    if input_freq != int(input_freq):
        raise ValueError("input_freq must be an integer")
    if output_freq != int(output_freq):
        raise ValueError("output_freq must be an integer")

    if input_freq == output_freq:
        new_trace = trace
    elif np.mod(input_freq, output_freq) == 0:
        new_trace = decimate(trace, q=input_freq//output_freq,
                             ftype='iir', zero_phase=True)
    else:
        new_trace = resample_poly(trace, up=output_freq, down=input_freq)
    return new_trace


def resize_to(x, new_length):
    x_new = np.zeros(new_length)
    length = np.shape(x)[0]
    if length >= new_length:
        extra = (length - new_length) // 2
        x_new = x[extra:new_length+extra]
    else:
        pad = (new_length - length) // 2
        x_new[pad:length+pad] = x
    return x_new


def convert_to_ndarray(string):
    elements = string.split(";")[:-1]
    ecg_samples = [int(x) for x in elements]
    return np.array(ecg_samples, dtype=np.float32)/512


def generate_signal(samples, gain, sensibility, sample_rate, n_samples=4096, final_sample_rate=400):
    x = convert_to_ndarray(samples)  # Convert string to Array
    x = x / gain * sensibility
    x = resample_ecg(x, sample_rate, final_sample_rate)  # Convert to 400 Hz
    x = resize_to(x, n_samples)  # Resize array
    x = 10 * x  # Rescale (Use 10 mV scale instead of 1mV scale)
    return x


def xml_to_ndarray(xml, n_samples=4096, final_sample_rate=400):
    n_leads = 12
    # Get information from exam
    ordered_dict = bf.data(fromstring(xml))['CONTEUDOEXAME']
    nexams = int(ordered_dict['QUANTIDADE']['$'])
    sample_rate = int(ordered_dict['TAXAAMOSTRAGEM']['$'].partition(' ')[0])
    sensibility = float(ordered_dict['SENSIBILIDADE']['$'].partition(' ')[0])
    register = ordered_dict['REGISTRO']

    # Get leads from first register
    # TODO: Pick reg from "dl_exame_tracado" (not critical: only 12 in ~2.5e9 exams were not the 1st)
    if nexams == 1:
        reg = register
    else:
        reg = register[0]
    leads = reg['DERIVACOES']['CANAL']
    x_all_leads = np.zeros((n_samples, n_leads), dtype=np.float32)
    for j in range(n_leads):
        lead = leads[j]
        x_all_leads[:, j] = generate_signal(samples=lead['AMOSTRAS']['$'],
                                            gain=lead['GANHO']['$'],
                                            sensibility=sensibility,
                                            sample_rate=sample_rate,
                                            final_sample_rate=final_sample_rate,
                                            n_samples=n_samples)

    return x_all_leads
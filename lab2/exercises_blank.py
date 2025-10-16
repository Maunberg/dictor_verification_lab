# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
import scipy.signal

from skimage.morphology import opening, closing


def load_vad_markup(path_to_rttm, signal, fs):
    # Function to read rttm files and generate VAD's markup in samples
    
    vad_markup = np.zeros(len(signal)).astype('float32')
        
    ###########################################################
    # Here is your code
    
    with open(path_to_rttm, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                # Формат RTTM: SPEAKER file 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
                start_time = float(parts[3])
                duration = float(parts[4])
                end_time = start_time + duration
    
                start_sample = int(start_time * fs)
                end_sample = int(end_time * fs)
                
                vad_markup[start_sample:end_sample] = 1.0
    
    ###########################################################
    
    return vad_markup

def framing(signal, window=320, shift=160):
    # Function to create frames from signal
    
    shape   = (int((signal.shape[0] - window)/shift + 1), window)
    frames  = np.zeros(shape).astype('float32')

    ###########################################################
    # Here is your code
    
    # Создаем фреймы с перекрытием
    for i in range(shape[0]):
        start_idx = i * shift
        end_idx = start_idx + window
        frames[i] = signal[start_idx:end_idx]
    
    ###########################################################
    
    return frames

def frame_energy(frames):
    # Function to compute frame energies
    
    # E = np.zeros(frames.shape[0]).astype('float32')
    # for i in range(frames.shape[0]):
    #     E[i] = np.sum(frames[i] ** 2)
    E = np.sum(frames ** 2, axis=1).astype('float32')
    
    return E

def norm_energy(E):
    # Function to normalize energy by mean energy and energy standard deviation
    
    E_norm = np.zeros(len(E)).astype('float32')

    ###########################################################
    # Here is your code
    
    # Вычисляем среднее и стандартное отклонение энергии
    mean_E = np.mean(E)
    std_E = np.std(E, ddof=1) # ddof=1 для вычисления несмещенной оценки стандартного отклонения
    
    # Нормализуем энергию (z-score нормализация)
    if std_E > 0:
        E_norm = (E - mean_E) / std_E
    else:
        E_norm = E - mean_E
    
    ###########################################################
    
    return E_norm

def gmm_train(E, gauss_pdf, n_realignment):
    # Function to train parameters of gaussian mixture model
    
    # Initialization gaussian mixture models
    w     = np.array([ 0.33, 0.33, 0.33])
    m     = np.array([-1.00, 0.00, 1.00])
    sigma = np.array([ 1.00, 1.00, 1.00])

    g = np.zeros([len(E), len(w)])
    for n in range(n_realignment):

        # E-step
        ###########################################################
        # Here is your code
        
        # Вычисляем апостериорные вероятности для каждого компонента
        for k in range(len(w)):
            for i in range(len(E)):
                g[i, k] = w[k] * gauss_pdf(E[i], m[k], sigma[k])
        
        # Нормализуем вероятности
        g_sum = np.sum(g, axis=1, keepdims=True)
        g = g / (g_sum + 1e-10)  # Добавляем малое значение для избежания деления на ноль

        ###########################################################

        # M-step
        ###########################################################
        # Here is your code
        
        # Обновляем веса компонентов
        w = np.mean(g, axis=0)
        
        # Обновляем средние значения
        for k in range(len(w)):
            if w[k] > 1e-10:  # Избегаем деления на ноль
                m[k] = np.sum(g[:, k] * E) / np.sum(g[:, k])
        
        # Обновляем дисперсии
        for k in range(len(w)):
            if w[k] > 1e-10:  # Избегаем деления на ноль
                sigma[k] = np.sqrt(np.sum(g[:, k] * (E - m[k])**2) / np.sum(g[:, k]))

        ###########################################################    
    return w, m, sigma

def eval_frame_post_prob(E, gauss_pdf, w, m, sigma):
    # Function to estimate a posterior probability that frame isn't speech

    g0 = np.zeros(len(E))

    ###########################################################
    # Here is your code
    
    # Вычисляем апостериорную вероятность для каждого фрейма
    for i in range(len(E)):
        # Вычисляем вероятности для всех компонентов
        probs = np.zeros(len(w))
        for k in range(len(w)):
            probs[k] = w[k] * gauss_pdf(E[i], m[k], sigma[k])
        
        # Нормализуем вероятности
        total_prob = np.sum(probs)
        if total_prob > 0:
            probs = probs / total_prob
        
        # g0 - это вероятность того, что фрейм НЕ является речью
        # Предполагаем, что первый компонент (индекс 0) соответствует не-речи
        g0[i] = probs[0]

    ###########################################################
            
    return g0

def energy_gmm_vad(signal, window, shift, gauss_pdf, n_realignment, vad_thr, mask_size_morph_filt):
    # Function to compute markup energy voice activity detector based of gaussian mixtures model
    
    # Squared signal
    squared_signal = signal**2
    
    # Frame signal with overlap
    frames = framing(squared_signal, window=window, shift=shift)
    
    # Sum frames to get energy
    E = frame_energy(frames)
    
    # Normalize the energy
    E_norm = norm_energy(E)
    
    # Train parameters of gaussian mixture models
    w, m, sigma = gmm_train(E_norm, gauss_pdf, n_realignment=10)
    
    # Estimate a posterior probability that frame isn't speech
    g0 = eval_frame_post_prob(E_norm, gauss_pdf, w, m, sigma)
    
    # Compute real VAD's markup
    vad_frame_markup_real = (g0 < vad_thr).astype('float32')  # frame VAD's markup

    vad_markup_real = np.zeros(len(signal)).astype('float32') # sample VAD's markup
    for idx in range(len(vad_frame_markup_real)):
        vad_markup_real[idx*shift:shift+idx*shift] = vad_frame_markup_real[idx]

    vad_markup_real[len(vad_frame_markup_real)*shift - len(signal):] = vad_frame_markup_real[-1]
    
    # Morphology Filters
    vad_markup_real = closing(vad_markup_real, np.ones(mask_size_morph_filt)) # close filter
    vad_markup_real = opening(vad_markup_real, np.ones(mask_size_morph_filt)) # open filter
    
    return vad_markup_real

def reverb(signal, impulse_response):
    # Function to create reverberation effect
    
    signal_reverb = np.zeros(len(signal)).astype('float32')
    
    ###########################################################
    # Here is your code
    
    # Применяем свертку сигнала с импульсной характеристикой для создания реверберации
    signal_reverb = scipy.signal.convolve(signal, impulse_response, mode='same')
    
    ###########################################################
    
    return signal_reverb

def awgn(signal, sigma_noise):
    # Function to add white gaussian noise to signal
    
    signal_noise = np.zeros(len(signal)).astype('float32')
    
    ###########################################################
    # Here is your code
    
    # Генерируем белый гауссов шум с заданным стандартным отклонением
    noise = np.random.normal(0, sigma_noise, len(signal))
    
    # Добавляем шум к исходному сигналу
    signal_noise = signal + noise
    
    ###########################################################
    
    return signal_noise
# Exercises in order to perform laboratory work

# Import of modules
import numpy as np
from math import sqrt
import itertools
import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow
from sklearn.metrics.pairwise import cosine_similarity
from common import get_eer


def get_tar_imp_scores(all_scores, all_labels):
    # Function to get target and impostors scores based on the labels

    tar_scores = []
    imp_scores = []
    for idx in range(len(all_labels)):

        if all_labels[idx] == 1:
            tar_scores.append(all_scores[idx])

        else:
            imp_scores.append(all_scores[idx])

    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)

    return tar_scores, imp_scores

def plot_histograms_2sets(all_scores_1, all_labels_1,
                          all_scores_2, all_labels_2,
                          names=['in-domain', 'out-of-domain']):
    # Function to show target/impostor histograms and compute EER to out-of-domain and in-domain datasets
    
    # Get target and impostors scores
    tar_scores_1, imp_scores_1 = get_tar_imp_scores(all_scores_1, all_labels_1)
    tar_scores_2, imp_scores_2 = get_tar_imp_scores(all_scores_2, all_labels_2)

    # Plot histograms for target and impostor scores
    min_scores = np.concatenate((tar_scores_1, tar_scores_2,
                                 imp_scores_1, imp_scores_2)).min()
    max_scores = np.concatenate((tar_scores_1, tar_scores_2,
                                 imp_scores_1, imp_scores_2)).max()

    hist(tar_scores_1, int(sqrt(len(tar_scores_1))), histtype='step', color='green',
         range=(min_scores, max_scores))
    hist(imp_scores_1, int(sqrt(len(imp_scores_1))), histtype='step', color='red',
         range=(min_scores, max_scores))
    hist(tar_scores_2, int(sqrt(len(tar_scores_2))), histtype='step', color='blue',
         range=(min_scores, max_scores))
    hist(imp_scores_2, int(sqrt(len(imp_scores_2))), histtype='step', color='cyan',
         range=(min_scores, max_scores))
    xlabel('$s$');
    ylabel('$\widehat{W}(s|H_0)$, $\widehat{W}(s|H_1)$');
    title('VoxCeleb1-O (cleaned), histograms');
    legend(list('{}_{}'.format(a[0], a[1]) for a in itertools.product(names, ['tar', 'imp'])))
    grid()
    show()

    # Compute equal error rate
    EER_1, thresh_EER_1 = get_eer(tar_scores_1, imp_scores_1)
    EER_2, thresh_EER_2 = get_eer(tar_scores_2, imp_scores_2)

    print("Equal Error Rate {0} (EER): {1:.3f}%, threshold EER: {2:.3f} ".format(names[0], EER_1, thresh_EER_1))
    print("Equal Error Rate {0} (EER): {1:.3f}%, threshold EER: {2:.3f} ".format(names[1], EER_2, thresh_EER_2))

def mean_embd_norm(test_embds, adapt_embds):
    # Function to apply mean embedding normalization
    
    test_embds_adapted = {}
    adapt_embds_list = [adapt_embds[k] for k in adapt_embds.keys()]
    mean_embd = torch.stack(adapt_embds_list).mean(0)
    if len(mean_embd.size()) > 1:
        mean_embd = mean_embd.mean(0)

    for k in test_embds.keys():
        test_embds_adapted[k] = test_embds[k] - mean_embd
    
    return test_embds_adapted

def s_norm(test_data, lines, adapt_data, N_s=200, eps=0.5):
    """
    Function to perform s-normalization for scores with the snorm_data
    :param test_data: test embeddings
    :param lines: test protocol
    :param scores: raw scores matrix
    :param adapt_data: data for s-norm (s-norm embeddings)
    :param N_s: top N impostors scrores for s-normalization
    :param eps: epsilon for std
    :return: snorm_scores - s-normalized scores
    """
    
    scores_adapted = []
    all_labels = []
    all_trials = []

    # Prepare lists of unique wavs from protocols
    enroll_list = list(set(list([x.strip().split()[1] for x in lines])))
    test_list   = list(set(list([x.strip().split()[2] for x in lines])))
    adapt_list  = list(adapt_data.keys())

    # Prepare entolls: save enroll embds in ndarray [num_wavs x emb_size]
    E = []
    for id, enr in enumerate(enroll_list):
        E.append(test_data[enr].squeeze(0).numpy())
    E = np.array(E)

    # Prepare tests: save test embds in ndarray [num_wavs x emb_size]
    T = []
    for id, tst in enumerate(test_list):
        T.append(test_data[tst].squeeze(0).numpy())
    T = np.array(T)

    # Prepare adapt data: save adapt embds in ndarray [num_wavs x emb_size]
    A = []
    for id, a in enumerate(adapt_list):
        A.append(adapt_data[a].squeeze(0).numpy())
    A = np.array(A)
    
    def l2_normalize(mat):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return mat / norms

    def stack_embeddings(keys, source_dict):
        emb_list = []
        for key in keys:
            emb_list.append(source_dict[key].squeeze(0).numpy())
        return np.stack(emb_list).astype(np.float32)

    E = l2_normalize(stack_embeddings(enroll_list, test_data))
    T = l2_normalize(stack_embeddings(test_list, test_data))
    A = l2_normalize(stack_embeddings(adapt_list, adapt_data))

    enroll_idx = {enroll_list[i]: i for i in range(len(enroll_list))}
    test_idx = {test_list[i]: i for i in range(len(test_list))}

    N_s = min(N_s, A.shape[0])

    enroll_stats = {}
    test_stats = {}

    def compute_stats(name, idx_map, emb_matrix, cache):
        if name in cache:
            return cache[name]

        vec = emb_matrix[idx_map[name]]
        sims = A.dot(vec)

        if N_s <= 0 or sims.size == 0:
            top_scores = sims
        elif N_s >= sims.size:
            top_scores = sims
        else:
            top_idx = np.argpartition(sims, -N_s)[-N_s:]
            top_scores = sims[top_idx]

        mu = float(np.mean(top_scores)) if top_scores.size else 0.0
        sigma = float(np.std(top_scores)) if top_scores.size else 0.0
        sigma = max(sigma, eps)

        cache[name] = (mu, sigma)
        return cache[name]

    for idx, line in enumerate(tqdm.tqdm(lines, total=len(lines), desc='Scoring with s-norm')):
        label, enroll_wav, test_wav = line.strip().split()

        enroll_vec = E[enroll_idx[enroll_wav]]
        test_vec = T[test_idx[test_wav]]

        raw_score = float(np.dot(enroll_vec, test_vec))

        mu_e, sigma_e = compute_stats(enroll_wav, enroll_idx, E, enroll_stats)
        mu_t, sigma_t = compute_stats(test_wav, test_idx, T, test_stats)

        score_norm = ((raw_score - mu_e) / (2.0 * sigma_e)) + ((raw_score - mu_t) / (2.0 * sigma_t))

        scores_adapted.append(score_norm)
        all_labels.append(int(label))
        all_trials.append(enroll_wav + " " + test_wav)

    return scores_adapted, all_labels, all_trials

class CalibrationDataset(Dataset):

    def __init__(self, target_scores, impostor_scores):
        super(CalibrationDataset, self).__init__()

        self.target_scores   = target_scores
        self.impostor_scores = impostor_scores
        self.L_tar = len(target_scores)
        self.L_imp = len(impostor_scores)

    def __len__(self):
        
        return 1

    def __getitem__(self, idx):

        return self.target_scores, self.impostor_scores

class LinearCalibrationModel(torch.nn.Module):
    # Building of the full model for constructing the extractor of features
    
    def __init__(self):
        super(LinearCalibrationModel, self).__init__()
        
        self.calib_params = nn.Linear(1, 1)

    def forward(self, x):
        
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        calib_x = self.calib_params(x)

        return calib_x
    
class CalibrationLoss(nn.Module):

    def __init__(self, ptar=0.01):
        '''
        Calibration loss. Code is based on https://github.com/alumae/sv_score_calibration/blob/master/calibrate_scores.py
        :param ptar: probability of target hypothesis
        '''
        
        super(CalibrationLoss, self).__init__()
        
        self.ptar  = ptar
        self.alpha = np.log(ptar/(1 - ptar))

    def forward(self, target_llrs, nontarget_llrs):

        def negative_log_sigmoid(lodds):
            # Function to compute -log(sigmoid(log_odds))
            
            return torch.log1p(torch.exp(-lodds))
        
        loss_value = 0
        
        target_llrs = target_llrs.view(-1)
        nontarget_llrs = nontarget_llrs.view(-1)

        tar_loss = negative_log_sigmoid(target_llrs + self.alpha)
        imp_loss = negative_log_sigmoid(-(nontarget_llrs + self.alpha))

        loss_value = self.ptar * tar_loss.mean() + (1 - self.ptar) * imp_loss.mean()

        return loss_value

def train_calibration(train_loader, model, criterion, optimizer, scheduler, num_epochs, verbose=False):
    # Function to train calibration model
    
    model.train()
    
    for epoch in range(0, num_epochs):
        
        loss = None
        for batch_idx, batch_data in enumerate(train_loader):
            tar_sc = batch_data[0]
            imp_sc = batch_data[1]
            
            tar_sc = tar_sc.squeeze(0).float().unsqueeze(1)
            imp_sc = imp_sc.squeeze(0).float().unsqueeze(1)

            optimizer.zero_grad()

            tar_scores = model(tar_sc)
            imp_scores = model(imp_sc)

            loss = criterion(tar_scores, imp_scores)
            loss.backward()
            optimizer.step()
                            
        lr_value = optimizer.param_groups[0]['lr']

        if verbose and loss is not None:
            print("Epoch {:1.0f}, LR {:f} Loss {:f}".format(epoch, lr_value, loss.item()))
                
        if scheduler is not None:
            scheduler[0].step()
    
    return
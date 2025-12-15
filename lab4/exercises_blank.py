# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow, colorbar


def tar_imp_hists(all_scores, all_labels):
    # Function to compute target and impostor histogram
    
    tar_scores = []
    imp_scores = []

    ###########################################################
    # Here is your code
    for score, label in zip(all_scores, all_labels):
        try:
            label_val = int(label)
        except (TypeError, ValueError):
            label_val = 1 if bool(label) else 0
        if label_val == 1:
            tar_scores.append(score)
        else:
            imp_scores.append(score)
    ###########################################################
    
    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)
    
    return tar_scores, imp_scores

def llr(all_scores, all_labels, tar_scores, imp_scores, gauss_pdf):
    # Function to compute log-likelihood ratio
    
    tar_scores_mean = np.mean(tar_scores)
    tar_scores_std  = np.std(tar_scores)
    imp_scores_mean = np.mean(imp_scores)
    imp_scores_std  = np.std(imp_scores)
    
    all_scores_sort   = np.zeros(len(all_scores))
    ground_truth_sort = np.zeros(len(all_scores), dtype='bool')
    
    ###########################################################
    # Here is your code
    all_scores = np.asarray(all_scores)
    all_labels = np.asarray(all_labels)

    sort_idx = np.argsort(all_scores)
    all_scores_sort = all_scores[sort_idx]
    ground_truth_sort = all_labels[sort_idx].astype(bool)
    ###########################################################
    
    tar_gauss_pdf = np.zeros(len(all_scores))
    imp_gauss_pdf = np.zeros(len(all_scores))
    LLR           = np.zeros(len(all_scores))
    
    ###########################################################
    # Here is your code
    eps = 1e-12
    tar_gauss_pdf = gauss_pdf(all_scores_sort, tar_scores_mean, tar_scores_std)
    imp_gauss_pdf = gauss_pdf(all_scores_sort, imp_scores_mean, imp_scores_std)

    tar_gauss_pdf = np.maximum(tar_gauss_pdf, eps)
    imp_gauss_pdf = np.maximum(imp_gauss_pdf, eps)

    LLR = np.log(tar_gauss_pdf / imp_gauss_pdf)
    ###########################################################
    
    return ground_truth_sort, all_scores_sort, tar_gauss_pdf, imp_gauss_pdf, LLR

def map_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar):
    # Function to perform maximum a posteriori test
    
    ground_truth_sort = np.asarray(ground_truth_sort, dtype=bool)
    LLR = np.asarray(LLR)

    sort_idx = np.argsort(LLR)
    LLR_sorted = LLR[sort_idx]
    gt_sorted = ground_truth_sort[sort_idx]

    len_thr = len(LLR_sorted)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    P_err   = np.zeros(len_thr)
    
    for idx in range(len_thr):
        thr_val = LLR_sorted[idx]
        solution = LLR_sorted > thr_val                                # decision
        
        err = (solution != gt_sorted)                                  # error vector
        
        fnr_thr[idx] = np.sum(err[ gt_sorted])/len(tar_scores)         # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~gt_sorted])/len(imp_scores)         # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        
        P_err[idx]   = fnr_thr[idx]*P_Htar + fpr_thr[idx]*(1 - P_Htar) # prob. of error
    
    # Plot error's prob.
    plot(LLR_sorted, P_err, color='blue')
    xlabel('$LLR$'); ylabel('$P_e$'); title('Probability of error'); grid(); show()
        
    P_err_idx = np.argmin(P_err) # argmin of error's prob.
    P_err_min = fnr_thr[P_err_idx]*P_Htar + fpr_thr[P_err_idx]*(1 - P_Htar)
    
    return LLR_sorted[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min

def neyman_pearson_test(ground_truth_sort, LLR, tar_scores, imp_scores, fnr):
    # Function to perform Neyman-Pearson test
    
    thr   = 0.0
    fpr   = 0.0
    
    ###########################################################
    # Here is your code
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)

    tar_count = len(tar_scores)
    imp_count = len(imp_scores)

    for idx in range(len_thr):
        solution = LLR > LLR[idx]
        err = solution != ground_truth_sort

        fnr_thr[idx] = np.sum(err[ground_truth_sort]) / tar_count
        fpr_thr[idx] = np.sum(err[~ground_truth_sort]) / imp_count

    idx_best = np.argmin(np.abs(fnr_thr - fnr))
    thr = LLR[idx_best]
    fpr = fpr_thr[idx_best]
    ###########################################################
    
    return thr, fpr

def bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar, C00, C10, C01, C11):
    # Function to perform Bayes' test
    
    thr   = 0.0
    fnr   = 0.0
    fpr   = 0.0
    AC    = 0.0
    
    ###########################################################
    # Here is your code
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    AC_thr  = np.zeros(len_thr)

    tar_count = len(tar_scores)
    imp_count = len(imp_scores)

    for idx in range(len_thr):
        solution = LLR > LLR[idx]
        err = solution != ground_truth_sort

        fnr_thr[idx] = np.sum(err[ground_truth_sort]) / tar_count
        fpr_thr[idx] = np.sum(err[~ground_truth_sort]) / imp_count

        AC_thr[idx] = (
            C00 * (1 - fnr_thr[idx]) * P_Htar +
            C10 * fnr_thr[idx] * P_Htar +
            C01 * fpr_thr[idx] * (1 - P_Htar) +
            C11 * (1 - fpr_thr[idx]) * (1 - P_Htar)
        )

    idx_best = np.argmin(AC_thr)

    thr = LLR[idx_best]
    fnr = fnr_thr[idx_best]
    fpr = fpr_thr[idx_best]
    AC  = AC_thr[idx_best]
    ###########################################################
    
    return thr, fnr, fpr, AC

def minmax_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar_thr, C00, C10, C01, C11):
    # Function to perform minimax test
    
    thr    = 0.0
    fnr    = 0.0
    fpr    = 0.0
    AC     = 0.0
    P_Htar = 0.0
    
    ###########################################################
    # Here is your code
    LLR = np.asarray(LLR)
    ground_truth_sort = np.asarray(ground_truth_sort, dtype=bool)

    len_thr = len(LLR)
    tar_count = len(tar_scores)
    imp_count = len(imp_scores)

    if len_thr == 0 or tar_count == 0 or imp_count == 0:
        return thr, fnr, fpr, AC, P_Htar

    sort_idx = np.argsort(LLR)
    LLR = LLR[sort_idx]
    ground_truth_sort = ground_truth_sort[sort_idx]

    gt_int = ground_truth_sort.astype(np.int32)
    imp_int = (~ground_truth_sort).astype(np.int32)

    cum_tar = np.cumsum(gt_int)
    cum_imp = np.cumsum(imp_int)

    fnr_thr = cum_tar / tar_count
    fpr_thr = (imp_count - cum_imp) / imp_count

    P_Htar_thr = np.asarray(P_Htar_thr)
    P_Htar_thr = np.asarray(P_Htar_thr)
    if P_Htar_thr.ndim == 0:
        P_Htar_thr = P_Htar_thr[np.newaxis]

    AC_surface = (
        (C00 * (1 - fnr_thr) + C10 * fnr_thr)[None, :] * P_Htar_thr[:, None] +
        (C01 * fpr_thr + C11 * (1 - fpr_thr))[None, :] * (1 - P_Htar_thr)[:, None]
    )

    best_idx = np.argmin(AC_surface, axis=1)
    AC_min_values = AC_surface[np.arange(len(P_Htar_thr)), best_idx]
    thr_values = LLR[best_idx]
    fnr_values = fnr_thr[best_idx]
    fpr_values = fpr_thr[best_idx]

    worst_idx = np.argmax(AC_min_values)

    thr = thr_values[worst_idx]
    fnr = fnr_values[worst_idx]
    fpr = fpr_values[worst_idx]
    AC = AC_min_values[worst_idx]
    P_Htar = P_Htar_thr[worst_idx]

    extent = [LLR.min(), LLR.max(), P_Htar_thr.min(), P_Htar_thr.max()]
    imshow(AC_surface, aspect='auto', origin='lower', extent=extent, cmap='magma')
    colorbar(label='$\\overline{C}$')
    plot(thr, P_Htar, 'ro')
    xlabel('$LLR$ threshold'); ylabel('$P(H_{tar})$')
    title('Average cost surface (top view)'); grid(False); show()
    ###########################################################
    
    return thr, fnr, fpr, AC, P_Htar
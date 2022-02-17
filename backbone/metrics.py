def precision(tp, fp):
    p = tp/(tp + fp)
    return p

def sensitivity(tp, fn):    # also called True Positive Rate
    s = tp/(tp + fn)
    return s

def FPR(fp, tn):            # False Positive Rate
    f = fp/(fp + tn)
    return f

def F1_score(precision, sensitivity, beta=2):
    f1 = (1 + beta**2) * ((precision * sensitivity)/(beta**2 * precision + sensitivity))
    return f1

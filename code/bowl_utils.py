import pandas as pd
import numpy as np
from numba import jit 
from functools import partial
import scipy as sp
import random
from sklearn.model_selection import train_test_split, StratifiedKFold


def get_train_valid_groups(train_groups, k, random_state, shuffle=True):
    train_groups_df = pd.DataFrame({'group':train_groups})
    g, g_len = zip(*train_groups_df['group'].value_counts().items())
    
    kfold = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=shuffle)
    train_index, _ = list(kfold.split(g, g_len))[k]    
    train_indices = np.array(g)[train_index]
    
    train_valid_bool = np.isin(train_groups, train_indices)    
         
    return train_valid_bool


def get_train_valid_rowids(train_samples, train_groups, k, random_state, random_state2=None, shuffle=True, choice=True):
    if random_state2 is None:
        random_state2 = random_state
    train_groups_df = pd.DataFrame({'group':train_groups})
    g, g_len = zip(*train_groups_df['group'].value_counts().items())
    
    kfold = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=shuffle)
    train_index, _ = list(kfold.split(g, g_len))[k]
    
    train_indices = np.array(g)[train_index]
    new_train_samples = [rowid for rowid, group in enumerate(train_groups) if group in train_indices]
    new_valid_samples = [rowid for rowid, group in enumerate(train_groups) if group not in train_indices]
    
    random.seed(random_state2)
    if choice:
        group_dict = {}
        for rowid, group in enumerate(train_groups):
            if group not in train_indices:
                if group not in group_dict:
                    group_dict[group] = []
                group_dict[group].append(rowid)
        new_valid_samples = [random.choice(v) for v in group_dict.values()]
    
    return np.array(new_train_samples), np.array(new_valid_samples)


def train_valid_split(train_samples, train_groups, k, random_state, random_state2=None, shuffle=True, choice=True):
    if random_state2 is None:
        random_state2 = random_state
    
    #print(f'random_state:{random_state}, random_state2:{random_state2}')
    train_groups_df = pd.DataFrame({'group':train_groups})
    g, g_len = zip(*train_groups_df['group'].value_counts().items())
    
    kfold = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=shuffle)
    train_index, _ = list(kfold.split(g, g_len))[k]
    
    train_indices = np.array(g)[train_index]
    new_train_samples = [row_id for row_id, group in zip(train_samples, train_groups) if group in train_indices]
    new_valid_samples = [row_id for row_id, group in zip(train_samples, train_groups) if group not in train_indices]    
    
    random.seed(random_state2)
    if choice:
        group_dict = {}
        for row_id, group in zip(train_samples, train_groups):
            if group not in train_indices:
                if group not in group_dict:
                    group_dict[group] = []
                group_dict[group].append(row_id)        
        new_valid_samples = [random.choice(v) for v in group_dict.values()]
    
    return np.array(new_train_samples), np.array(new_valid_samples)


@jit
def qwk3(a1, a2, max_rat=3):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / (e+1e-08)



class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])

        return -qwk3(y, X_p)

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']


def get_optimized_kappa_score(predictions, groundtruth):
    optR = OptimizedRounder()
    optR.fit(predictions, groundtruth)
    coefficients = optR.coefficients()
    #print(coefficients)
    temp_predictions = predictions.copy()
    temp_predictions[temp_predictions < coefficients[0]] = 0
    temp_predictions[(coefficients[0]<=temp_predictions)&(temp_predictions< coefficients[1])] = 1
    temp_predictions[(coefficients[1]<=temp_predictions)&(temp_predictions< coefficients[2])] = 2
    temp_predictions[(coefficients[2]<=temp_predictions)] = 3

    kappa_score = qwk3(temp_predictions, groundtruth)
    return kappa_score

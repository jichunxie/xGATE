import numpy as np
import warnings
import pandas as pd
import statsmodels.api as sm
from scipy.sparse import issparse
from statsmodels.regression.quantile_regression import QuantReg
from scipy.stats import norm

class SiFiNet:
    def __init__(self, data, sparse, meta_data, gene_name, data_name, n, p, data_thres=None, coexp=None, est_ms=None, thres=0, q5=None, kset=None, conn=None, conn2=None, fg_id=None, uni_fg_id=None, uni_cluster=None, selected_cluster=None, featureset=None):
        self.data = data
        self.sparse = sparse
        self.meta_data = meta_data
        self.gene_name = gene_name
        self.data_name = data_name
        self.n = n
        self.p = p
        self.data_thres = data_thres
        self.coexp = coexp
        self.est_ms = est_ms
        self.thres = thres
        self.q5 = q5
        self.kset = kset
        self.conn = conn
        self.conn2 = conn2
        self.fg_id = fg_id
        self.uni_fg_id = uni_fg_id
        self.uni_cluster = uni_cluster
        self.selected_cluster = selected_cluster
        self.featureset = featureset

    def update_data_thres(self, data_thres):
        self.data_thres = {'dt': data_thres}  # Use a dictionary to mimic R's list structure
        self.data_thres['name'] = self.data_name
    
    def set_coexp(self, coexp_matrix):
        self.coexp = pd.DataFrame(coexp_matrix)
        for i in range(len(self.coexp)):
            self.coexp.iat[i, i] = 0
    
    def update_dt(self, dt):
        self.dt = dt

def create_sifinet_object(counts, gene_name=None, meta_data=None, data_name=None, sparse=False, rowfeature=True):
    if rowfeature:
        counts = counts.T  # Transpose the DataFrame if rows are features

    if gene_name is None:
        if hasattr(counts, 'columns'):
            gene_name = counts.columns.tolist()
        else:
            gene_name = list(range(1, counts.shape[1] + 1))

    #if data_name is None:
        #data_name = "data1"

    if meta_data is None:
        meta_data = pd.DataFrame(np.zeros((counts.shape[0], 0)))

    if issparse(counts):
        q5 = np.percentile(counts.toarray(), 50, axis=0)  # 50% quantile for each column
    else:
        q5 = counts.quantile(0.5).tolist()  # 50% quantile for each column

    n = counts.shape[0]
    p = counts.shape[1]

    object = SiFiNet(
        data=counts,
        sparse=sparse,
        meta_data=meta_data,
        gene_name=gene_name,
        data_name=data_name,
        n=n,
        p=p,
        q5=q5,
        kset=list(range(p)),
        featureset={'unique': [], 'shared': [], 'enriched': []}
    )
    return object

def quantile_thres(so, M_full, M_subcohort, sparse=False):
    M_full = pd.DataFrame.transpose(M_full)
    M_subcohort = pd.DataFrame.transpose(M_subcohort)
    n, p = M_full.shape  # dimensions from full matrix
    Z = np.mean(M_full, axis=1)  # Use colMeans from the full matrix

    if not so.sparse:
        dt = np.zeros((n, p))
        for j in range(p):
            temp = so.data.iloc[:, j].values
            #print("temp:", temp)
            #print("Z:", Z.flatten())
            v5 = np.quantile(temp, 0.5)
            #print("v5:", v5)
            if (v5 == 0) or (v5 >= np.quantile(temp, 0.99)):
                dt[:, j] = (temp > 0).astype(int)
            else:
                quant = np.sum(temp <= v5) / n
                #print("quant:", quant)
                
                # Fit quantile regression using statsmodels
                model = sm.QuantReg(temp, sm.add_constant(Z))
                res = model.fit(q=quant)
                Q = res.fittedvalues
                #print("Q:", Q)
                
                dt[:, j] = (temp > Q).astype(int)

    # Filter dt to only include rows that are common with M_subcohort
    common_indices = M_full.index.isin(M_subcohort.index)
    filtered_dt = dt[common_indices, :]

    # Update the 'data_thres' attribute for the 'so' object with the filtered data
    so.update_data_thres(filtered_dt)
    so.update_dt(dt)
    return so

def quantile_thres2(so):
    n = so.n
    p = so.p
    Z = np.mean(so.data, axis=1).values  # Convert Series to NumPy array


    if not so.sparse:
        dt = np.zeros((n, p))
        for j in range(p):
            temp = so.data.iloc[:, j].values
            #print("temp:", temp)
            #print("Z:", Z.flatten())
            v5 = np.quantile(temp, 0.5)
            #print("v5:", v5)
            if (v5 == 0) or (v5 >= np.quantile(temp, 0.99)):
                dt[:, j] = (temp > 0).astype(int)
            else:
                quant = np.sum(temp <= v5) / n
                #print("quant:", quant)
                
                # Fit quantile regression using statsmodels
                model = sm.QuantReg(temp, sm.add_constant(Z))
                res = model.fit(q=quant)
                Q = res.fittedvalues
                #print("Q:", Q)
                
                dt[:, j] = (temp > Q).astype(int)

    so.update_data_thres(dt)        
    return so

def norm_FDR_SQAUC(value, sam_mean, sam_sd, alpha, n, p):
    tp = 2 * np.sqrt(np.log(max(n, p))) * sam_sd
    d = len(value)
    value_a = np.abs(value - sam_mean)
    value_ord = np.argsort(-value_a)  # Descending order

    value_a_s = value_a[value_ord]
    P22s = 2 * (1 - norm.cdf(value_a_s, loc=0, scale=sam_sd))  # Two-tailed test
    FDR2h = P22s * d / np.arange(1, d + 1)
    R2 = np.max(np.where(FDR2h <= alpha)[0]) if np.any(FDR2h <= alpha) else -1

    if R2 == -1:
        return tp
    elif value_a_s[R2] > tp:
        return tp
    else:
        return value_a_s[R2]
    
def feature_coexp(self):
    if not self.sparse:
        self.coexp = self.cal_coexp(self.data_thres[0])
    else:
        self.coexp = self.cal_coexp_sp(self.data_thres[0])

    np.fill_diagonal(self.coexp, 0)
    return self

def filter_lowexp(so, t1=10, t2=0.9, t3=0.9):
    if so.coexp is None or so.est_ms is None:
        raise ValueError("Coexpression matrix and estimated means/stds must be set before filtering.")

    r_set = set()
    for target in range(3):  # Assuming q5 values are discretized as 0, 1, 2
        screen_set = np.where(so.q5 == target)[0]
        if len(screen_set) >= 2:
            abs_sum = np.sum(np.abs(so.coexp[:, screen_set] - so.est_ms['mean']) > so.thres, axis=0)
            pos_sum_w = np.sum(so.coexp[np.ix_(screen_set, screen_set)] > (so.thres + so.est_ms['mean']), axis=0)
            pos_sum = np.sum(so.coexp[:, screen_set] > (so.thres + so.est_ms['mean']), axis=0)
            
            valid_indices = screen_set[(abs_sum >= t1) & (pos_sum / abs_sum >= t2) & (pos_sum_w / pos_sum >= t3)]
            r_set.update(valid_indices)

    # Update the kset to exclude filtered genes
    so.kset = list(set(range(so.p)) - r_set)
    return so

# This version of EstNull makes two changes: stops at t=2000 instead of t=1000, and sets uhat and shat values to their final values if the break condition is not reached
def EstNull(x, gamma=0.1):
    n = len(x)
    gan = n ** -gamma
    #print(f"n: {n}, gamma: {gamma}, gan: {gan}")
    
    shat = 0
    uhat = 0

    for t in range(1, 1001):
        s = t / 200.0
        phiplus = np.mean(np.cos(s * x))
        phiminus = np.mean(np.sin(s * x))
        phi = np.sqrt(phiplus**2 + phiminus**2)
        #print(f"t: {t}, s: {s}, phiplus: {phiplus}, phiminus: {phiminus}, phi: {phi}")
        
        if phi <= gan or t==1000:
            if t == 1000 and phi > gan:
                warnings.warn("Reached the maximum iteration without meeting the condition phi <= gan. Adjacency matrix quality is not optimal.")
            dphiplus = -np.sum(x * np.sin(s * x)) / n
            dphiminus = np.sum(x * np.cos(s * x)) / n
            shat = np.sqrt(- (phiplus * dphiplus + phiminus * dphiminus) / (s * phi * phi))
            uhat = -(dphiplus * phiminus - dphiminus * phiplus) / (phi * phi)
            #print(f"Condition met at t: {t}")
            #print(f"dphiplus: {dphiplus}, dphiminus: {dphiminus}, shat: {shat}, uhat: {uhat}")
            break
    
    #print(f"Final shat: {shat}, uhat: {uhat}")
    return {'mean': uhat, 'std': shat}
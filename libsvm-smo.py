import numpy as np
import pandas as pd

data_total = pd.read_csv('./ESL-data/ESLmixture.csv').iloc[:, 1:]

train_data = data_total.iloc[:, [0, 1]]
label = data_total["label"].copy()
label[data_total["label"] == 0] = -1

num_data, num_feature = train_data.shape

# Initialize some parameters
iter_num = 50
C = 0.6
alpha_vec = np.zeros((num_data,1))
G_vec = np.ones((num_data,1)) * -1
tau = 1e-3

def selectB():
    i = -1
    G_max = -np.inf
    G_min = np.inf
    for t in range(num_data):
        if ((label[t] == 1) and (alpha_vec[t] < C)) or\
                ((label[t] == -1) and alpha_vec[t] > 0):
            if (-label[t]*G_vec[t] >= G_max):
                i = t
                G_max = -label[t]* G_vec[t]

    j = -1
    obj_min = np.inf
    for t in range(num_data):
        if ((label[t] == 1) and (alpha_vec[t] > 0)) or \
                ((label[t] == -1) and alpha_vec[t] < 0):
            b = G_max + label[t]*G_vec[t]

            if (-label[t] * G_vec[t] <= G_min):
                G_min = -label[t]*G_vec[t]
            if (b > 0):
                a = Q[i,i] + Q[t,t] - 2*label[i]*label[t]*Q[i,t]
                if (a <= 0):
                    a = tau
                if (-(b*b)/a <= obj_min):
                    j = t
                    obj_min = -(b*b)/a

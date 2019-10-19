import numpy as np
import pandas as pd

# data_total = pd.read_csv('./ESL-data/ESLmixture.csv').iloc[:, 1:]
#
# train_data = data_total.iloc[:, [0, 1]]
# label = data_total["label"].copy()
# label[data_total["label"] == 0] = -1
#
# num_data, num_feature = train_data.shape
#
# # Initialize some parameters
# C = 0.01

class SVMClassifer():
    def __init__(self, train_data, label, penalty):
        self.train_data = train_data
        self.label = label.copy()
        self.label[label==0] = -1
        self.penalty = penalty
        self.alpha_vec, self.w, self.intercept = self.SVM_SMO(self.train_data, self.label, self.penalty)
        self.prediction = self.predict()

    def predict(self):
        prediction = np.multiply(self.alpha_vec, self.label[:,np.newaxis]).T
        prediction = np.dot(prediction, np.inner(self.train_data, self.train_data)) + self.intercept
        self.prediction = np.sign(prediction)

    def decision_bound(self, test_data):
        pred = self.w * test_data[:,0] + self.intercept
        self.ret_label = np.zeros((test_data.shape[0], 1))
        self.ret_label[test_data[:,1] > pred] = 1
        self.ret_label[test_data[:,1] <= pred] = -1
        return (self.ret_label)

    def find_new_index(self, i,num_data):
        j = np.random.randint(0, num_data, 1)
        if (i == j):
            return (self.find_new_index(i,num_data))
        else:
            return (j)

    # main function for svm-smo
    def SVM_SMO(self, train_data, label, C):
        num_data, num_feature = train_data.shape
        iter_i = 0
        intercept = 0
        alpha_vec = np.zeros((num_data, 1))
        iter_num = 1000
        while iter_i <= iter_num:
            alpha_changed = 0
            for i in range(num_data):
                gX_i = (np.inner((alpha_vec.reshape(-1) * label),
                                (np.dot(train_data, train_data.iloc[i, :].T)))) + intercept
                E_i = gX_i - (label[i])
                if (((label[i] * E_i < -1e-3) and (alpha_vec[i] < C))
                        or ((label[i] * E_i > 1e-3) and (alpha_vec[i] > 0))):
                    j = int(self.find_new_index(i,num_data))
                    gX_j = (np.inner((alpha_vec.reshape(-1) * label),
                                    (np.dot(train_data, train_data.iloc[j, :].T)))) + intercept
                    E_j = gX_j - (label[j])
                    alpha_old_i = alpha_vec[i].copy()
                    alpha_old_j = alpha_vec[j].copy()

                    if label[i] != label[j]:
                        L = np.maximum(0, alpha_old_j - alpha_old_i)
                        H = np.minimum(C, C + alpha_old_j - alpha_old_i)
                    else:
                        L = np.maximum(0, alpha_old_j + alpha_old_i - C)
                        H = np.minimum(C, alpha_old_j + alpha_old_i)

                    if L == H:
                        continue

                    x1 = train_data.iloc[i, :]
                    x2 = train_data.iloc[j, :]
                    eta = 2 * np.inner(x1, x2) - np.inner(x1, x1) - np.inner(x2, x2)
                    alpha_new_j = alpha_old_j - label[j] * (E_i - E_j) / eta

                    if alpha_new_j > H:
                        alpha_new_j = H
                    elif alpha_new_j < L:
                        alpha_new_j = L

                    alpha_vec[j] = alpha_new_j
                    if np.abs((alpha_new_j - alpha_old_j)) < 1e-6:
                        continue
                    alpha_new_i = alpha_old_i + label[i] * label[j] * (alpha_old_j - alpha_new_j)

                    # intercept decision
                    intercept_new_1 = (intercept - E_i - label[i] * (alpha_new_i - alpha_old_i) * np.inner(x1, x1)
                                    - label[j] * (alpha_new_j - alpha_old_j) * np.inner(x1, x2))
                    intercept_new_2 = (intercept - E_j - label[i] * (alpha_new_i - alpha_old_i) * np.inner(x1, x2)
                                    - label[j] * (alpha_new_j - alpha_old_j) * np.inner(x2, x2))

                    if (alpha_new_i > 0) and (alpha_new_i < C):
                        intercept = intercept_new_1
                    elif (alpha_new_j > 0) and (alpha_new_j < C):
                        intercept = intercept_new_2
                    else:
                        intercept = (intercept_new_1 + intercept_new_2) / 2
                    alpha_vec[i] = alpha_new_i
                    alpha_changed += 1

            if alpha_changed == 0:
                iter_i += 1
            else:
                iter_i = 0

        W = np.dot(np.multiply(alpha_vec, label[:,np.newaxis]).T, train_data)
        W = W[0]
        w = -W[0]/W[1]
        print(intercept)
        intercept = intercept/W[1]
        print(intercept)
        return(alpha_vec, w, intercept)

# alpha_vec, intercept = SVM_SMO(train_data,label,C)
# W = np.dot(np.multiply(alpha_vec, label[:,np.newaxis]).T, train_data)
# W = W[0]
# k = -W[0]/W[1]
# print(k)
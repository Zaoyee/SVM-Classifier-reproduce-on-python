import numpy as np
import pandas as pd

data_total = pd.read_csv('./ESL-data/ESLmixture.csv').iloc[:, 1:]

train_data = data_total.iloc[:, [0, 1]]
label = data_total["label"].copy()
label[data_total["label"] == 0] = -1

num_data, num_feature = train_data.shape

# Initialize some parameters
alpha_vec = np.zeros((num_data, 1))
iter_num = 40
C = 100


def find_new_index(i):
    j = np.random.randint(0, num_data, 1)
    if (i == j):
        return (find_new_index(i))
    else:
        return (j)


# main function for svm-smo
def SVM_SMO(train_data, label, C):
    iter_i = 0
    intercept = 0
    while iter_i <= iter_num:
        alpha_changed = 0
        for i in range(num_data):
            gX_i = float(np.inner((alpha_vec.reshape(-1) * label),
                            (np.dot(train_data, train_data.iloc[i, :].T)))) + intercept
            E_i = gX_i - float(label[i])
            if (((label[i] * E_i < -1e-3) and (alpha_vec[i] < C))
                    or ((label[i] * E_i > 1e-3) and (alpha_vec[i] > 0))):
                j = int(find_new_index(i))
                gX_j = float(np.inner((alpha_vec.reshape(-1) * label),
                                (np.dot(train_data, train_data.iloc[j, :].T)))) + intercept
                E_j = gX_j - float(label[j])
                alpha_old_i = alpha_vec[i]
                alpha_old_j = alpha_vec[j]

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
                if np.abs(alpha_new_j - alpha_old_j) < 1e-4:
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
    return(alpha_vec, intercept)
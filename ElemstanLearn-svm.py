import pandas as pd
import numpy as np
import seaborn as sns; sns.set_style("white")
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import svmSmo as ss
from scipy.stats import multivariate_normal


"""
Read bunch of data over here
"""
data_total = pd.read_csv('./ESL-data/ESLmixture.csv').iloc[:,1:]

px1 = pd.read_csv('./ESL-data/ESLpx1.csv').iloc[:,1]
px2 = pd.read_csv('./ESL-data/ESLpx2.csv').iloc[:,1]

prob = np.array(pd.read_csv('./ESL-data/ESLprob.csv').iloc[:,1].values).reshape(len(px2),len(px1))
prob_list = pd.read_csv('./ESL-data/ESLprob.csv').iloc[:,1]

marginal = pd.read_csv('./ESL-data/ESLmarginal.csv').iloc[:,1]
means = pd.read_csv('./ESL-data/ESLmeans.csv').iloc[:,[1,2]]
xnew = pd.read_csv('./ESL-data/ESLxnew.csv').iloc[:,[1,2]]

"""
Compute the Bayes error
"""
prob_list[prob_list >= .5] = 1 - prob_list[prob_list >= .5].values
bayes_error = np.sum(marginal*prob_list)

data_total['fold'] = None
data_total['color'] = None
np.random.seed(123)
data_total['fold'] = np.random.randint(1, 11, size=(data_total.shape[0],1))

# define the svm classifer model
def svc_model(penalty):
    """
    The model that in the scikitlearn package
    """
    model = SVC(kernel='linear', C=penalty)
    model.fit(data_total.iloc[:,[0,1]], data_total['label'])
    return(model)

def svc_smo_scrath(penalty):
    """
    Create the model class that I write
    """
    model = ss.SVMClassifer(train_data=data_total.iloc[:,[0,1]],
                            label=data_total['label'],
                            penalty=penalty)
    return(model)

def abline(ax, slope, intercept, type):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, type, c='k')
    
def tst_error(model):
    """
    This function will compute the test error on the model I write in scratch
    """
    pdf_mat = np.zeros((xnew.shape[0],means.shape[0]))
    pdf_mat = pd.DataFrame(pdf_mat)
    
    for i in range(means.shape[0]):
        supermean = means.iloc[i,:]
        pdf_mat.iloc[:,i] = multivariate_normal.pdf(x=xnew, mean=supermean)
        
    x_new_label = pdf_mat.idxmax(axis=1)
    pred_pdf = x_new_label.copy()
    x_new_label[pred_pdf>9] = 1
    x_new_label[pred_pdf<=9] = -1
    
    predict_grid_label = model.decision_bound(np.array(xnew))
    predict_grid_label[predict_grid_label==0] = -1
    tst_error = np.sum(predict_grid_label.reshape(-1) != x_new_label) / x_new_label.shape[0]
    #plt.scatter(xnew.iloc[:,0],xnew.iloc[:,1],c=x_new_label)
    return(tst_error)
    

def plot_svc(model, ax=None, mode='package'):
    """
    This function will do all steps to plot the similar figure
    ,such as support_vectors, contour line for margins, bacakground dots and 
    anything related. 
    """
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    PX1, PX2 = np.meshgrid(px1, px2)
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    # grid
    if mode=='package':
        """The plot in scikitlearn mode"""
        P = model.decision_function(xy).reshape(X.shape)
        grid_label = model.predict(xy)
        grids = np.hstack([xy, grid_label[:,np.newaxis]])
        grids = pd.DataFrame(grids)
        grids.columns = ['X1', 'X2', 'label']
        #boudary and margins
        ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1],
                   alpha=0.8, linestyles=['--', '-', '--'])
        #beta = model.coef_[0]
        # w = -beta[0] / beta[1]
        # xx = np.linspace(xlim[0], xlim[1])
        # intercept = model.intercept_ / beta[1]
        ax.scatter(x="X1", y='X2', c='label', data=grids, cmap=cmap, s=0.1)
        # compute the margin
        # margin = 1 / np.sqrt(np.sum(model.coef_ ** 2))
        # margin = np.sqrt(1 + w ** 2) * margin
        alphas = np.abs(model.dual_coef_)
        support_index = np.where((alphas > 0) & (alphas < 1e4))[1]
        train_error = 1 - model1.score(data_total.iloc[:,[0,1]], data_total['label'])
        
        for i in support_index:
            ax.scatter(x=model.support_vectors_[i,0], y=model.support_vectors_[i,1],
                       c='black', linewidth=2)
    elif mode == 'scratch':
        """The plot in my own code mode-SMO"""
        grid_label = model.decision_bound(xy)
        gd = np.hstack([xy, grid_label])
        gd = pd.DataFrame(gd)
        gd.columns = ['X1', 'X2', 'label']
        ax.scatter(x="X1", y='X2', c='label', data=gd, cmap=cmap, s=0.1)

        alphas = model.alpha_vec
        support_index = np.where((alphas > 0) & (alphas < model.penalty))[0]
        for i in support_index:
            ax.scatter(x=model.train_data.iloc[i,0], y=model.train_data.iloc[i,1],
                       c='black', linewidth=2)
        train_error = model.train_error()
        
        abline(ax, model.w, -model.intercept, '-')
        abline(ax, model.w, -model.intercept+model.margin, '--')
        abline(ax, model.w, -model.intercept-model.margin, '--')
        
        
    """Create the label annotation here"""
    ax.contour(PX1, PX2, prob, levels=[.5],
               colors='purple', linestyles='--', alpha=0.9)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xlim((np.min(px1)-0.1, np.max(px1)+0.1))
    ax.set_ylim((np.min(px2)-0.1, np.max(px2)+0.1))
    ax.annotate('Train Error  :  {:.3f}\nTrain Error  :  {:.2f}\nBayes Error  :  {:.2f}'.format(train_error,0,bayes_error), ha='left', va='top',
                xy=(0.01, 0.08), xycoords="axes fraction",
                bbox=dict(boxstyle='round', fc='w'), size=12)


"""
The generation of left figure

"""
plt.figure(figsize=(8,8))
model1 = svc_model(1e4)
colors = [(0.1500, 0.4667, 0.7059, 0.7),
          (1.000, 0.6, 0.002, 0.5)]
cmap = ListedColormap(colors)
plt.scatter(x="X1", y="X2", c="label", data=data_total, cmap=cmap, linewidth=2)
plot_svc(model1,mode='package')
plt.xlabel('C = 10000', fontsize=13)
plt.savefig('./Figs/out1.png')

"""
The generation of second figure

"""
# This part is working on the algorithm coded from the scratch
model2 = svc_smo_scrath(1e-2)
plt.figure(figsize=(8,8))
colors = [(0.1500, 0.4667, 0.7059, 0.7),
          (1.000, 0.6, 0.002, 0.5)]
cmap = ListedColormap(colors)
plt.scatter(x="X1", y="X2", c="label", data=data_total, cmap=cmap, linewidth=2)
plot_svc(model2, mode='scratch')
plt.xlabel('C = 0.01', fontsize=13)
plt.savefig('./Figs/out2.png')


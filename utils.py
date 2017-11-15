from sklearn import datasets, model_selection
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.legend_handler import HandlerLine2D
from utils import *


def scatter_dataset(ax,X,y,s=10,**params):
    for i in range(int(np.max(y))+1):
        ax.scatter(X[y==i][:,0],X[y==i][:,1],s=s,**params)
        ax.scatter(X[y==i][:,0],X[y==i][:,1],s=s,**params)
      
def plot_contours(ax, predict, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = predict(np.c_[xx.ravel(), yy.ravel()])
#     Z[Z<0]=0
#     Z[Z>0]=1
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, int(np.max(Z)),**params)
    return out

def make_meshgrid(x_min,x_max,y_min,y_max, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy    

def spiral(n):
    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    Y=[]
    X = []
    for i in range(4):
        theta = np.random.uniform(low=4.5,high=6.5,size=n)
        rho = 50 + 2 * theta**3 + np.random.randn(n)*20
        x,y = pol2cart(rho,theta+i*np.pi/2)
        A=np.stack((x,y))
        X.append(A.T)
        lbl = np.ones(n)*i
        Y.append(lbl)
    X = np.concatenate(X)
    X /= np.max(abs(X))
    return X, np.concatenate(Y)

def gen_examples(batch,dataset,epsilon=0.006,dim=2):
    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    
    x = np.random.uniform(-1.2,1.2,size=(batch,dim))
    
    for j in range(batch):
        x1 = x[j,0] * np.ones(dataset.shape[0])
        x2 = x[j,1] * np.ones(dataset.shape[0])
        e = (x1-dataset[:,0])**2+(x2-dataset[:,1])**2
        if np.min(e)<epsilon:
            x[j,0]=0
            x[j,1]=0
        
    return x
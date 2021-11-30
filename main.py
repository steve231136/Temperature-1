import numpy as np


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C

from sklearn.gaussian_process.kernels import WhiteKernel as wh

from sklearn.gaussian_process.kernels import ExpSineSquared as expsin

from sklearn.gaussian_process.kernels import DotProduct as dt


from sklearn.gaussian_process.kernels import Exponentiation as Exp

from sklearn.gaussian_process.kernels import RationalQuadratic as RQ

np.random.seed(1)

df = pd.read_csv('MTemp.csv', sep =';')

df_array = np.asarray(df)

time= df_array[1:14,0]
tim = time.astype(float)


humidity = df_array[1:14,1]
humid= humidity.astype(float)

indice = df_array[1:14,2]
ind = indice.astype(float)

temperature = df_array[1:14,3]
temp = temperature.astype(float)


y = np.array([temp]).T

print('y=',y)

X= np.array([tim,humid]).T
X1 = np.atleast_2d([tim]).T

print('X=',X)

t = np.linspace(0,50,100).T

tp =np.array([np.linspace(0,50,100),np.linspace(50,100,100)]).T
#Data for which we need predictions(tp)

print('tp=',tp)

kernel1= C()*RQ()

print('kernel=', kernel1)
gp = GaussianProcessRegressor(kernel= kernel1, n_restarts_optimizer=80).fit(X,y)


y_pred_1, sigma1 =gp.predict(tp, return_std=True)

print(y_pred_1)


plt.plot(t,y_pred_1 + 1.96*sigma1, alpha = 0.3, color='k')
  
plt.plot(t,y_pred_1 - 1.96*sigma1, alpha = 0.3, color='k')


plt.plot(t,y_pred_1,'b.',markersize=5 ,label = u'predicted')


plt.plot(tim,y,'r.',markersize=2,label = u'measured')

plt.show()

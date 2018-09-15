from statsmodels.tsa.arima_model import ARIMA
from statsmodels import robust
import numpy as np
import pandas as pd
from random import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
# contrived dataset
csv=pd.read_csv('C:\\Users\\HP\\Downloads\\wolfs-sunspot-numbers-1700-1988.csv')
data=csv['Values']
data=np.asarray(data)
data_train=data[:220]
data_test=data[220:]
datay=np.asarray(csv['Year'])
datay_train=datay[:220]
datay_test=datay[220:]
print(data)
history = [x for x in data_train]
plt.plot(datay,data)
plt.show()
# fit model
data_pred=[]
data_train1=[]
for i in range(len(data_test)):
    j=220+i
    data_train1.append(data[:j])
print(history)
for t in range(len(data_test)):
    j=220+t
    if j>=253 and j<=255:
        j=252
    model = ARIMA(data[:j], order=(4,0,4))
    model_fit = model.fit(disp=False)
    output = model_fit.forecast()
    yhat = output[0]
    obs = data_test[t]
    data_pred=np.append(data_pred,yhat)
    history.append(obs)
model = ARIMA(data_train, order=(4,0,4))
model_fit = model.fit(disp=False)
# make prediction
y_hat = model_fit.predict(start=4,end=len(data_train))
y_hat1 = np.append(y_hat,data_pred)
print(data_pred,'\n\n\n\n')
plt.title('ARIMA Model (4,0,4)')
plt.ylabel('Values')
plt.xlabel('Year')
plt.plot(datay,data,'r',marker='o',label='original')
plt.plot(datay[3:],y_hat1,'b',marker='^',label='ARIMA(4,0,4)')
plt.legend(loc='upper right')
MSE=mse(data_train[3:],y_hat)
MAD=robust.mad(y_hat)
R2=r2(data_train[3:],y_hat)
print (y_hat)
print ('\n\n\n\n\n')
print (data[3:])
print ('MSE(Train)=',MSE,'MAD(Train)=',MAD,'R2(Train)=',R2)
MSE_test=mse(data_test,data_pred)
MAD_test=robust.mad(data_pred)
R2_test=r2(data_test,data_pred)
print ('MSE(Test)=',MSE_test,'MAD(Test)=',MAD_test,'R2(Test)=',R2_test)
MSE_test=mse(data[3:],y_hat1)
MAD_test=robust.mad(y_hat1)
R2_test=r2(data[3:],y_hat1)
print ('MSE(Whole)=',MSE_test,'MAD(Whole)=',MAD_test,'R2(Whole)=',R2_test)
plt.grid()
plt.show()

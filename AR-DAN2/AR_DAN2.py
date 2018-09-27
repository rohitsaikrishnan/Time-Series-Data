import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from pyramid.arima import auto_arima
import pandas as pd
import numpy as np
from sklearn import linear_model
# contrived dataset
csv=pd.read_csv('C:\\Users\\HP\\Downloads\\wolfs-sunspot-numbers-1700-1988.csv')
data=csv['Values']
datay=np.asarray(csv['Year'])
data=np.asarray(data)
data_train=data[:220]#Train Data
datay_train=datay[:220]
data_test=data[220:]#Test Data
datay_test=datay[220:]
model = auto_arima(data_train, trace=True, error_action='ignore', suppress_warnings=True)#Running Auto_Arima
model.fit(data_train)
y_hat = model.predict_in_sample()#Fitting the model on training data
MSE_test=mse(data_train,y_hat)#Calculating MSE of ARIMA
MSE_test_before=MSE_test


'''
Running the AR-DAN2 from this step
'''
k=1000#max_iterations
MSE_min=120#Minimum MSE
ALL_MSE=[]
MSE_test=mse(data_train,y_hat)#Calculate MSE for Step 0
ALL_MSE.append(MSE_test)
parameter_tuples=[]
print("MSE before iteration:",MSE_test)
print('MSE a b')
y_hat=y_hat.reshape(-1,1)
y_temp=[]
for i in range(k):
    regr=linear_model.LinearRegression()#Linear Regression Object
    '''PLEASE LOOK INTO THE NEXT STEP.THIS COULD BE CRUCIAL'''
    regr.fit(y_hat,data_train)#Fitting Linear Regression model for establishing relationship between y(i),original_data.
    '''THE LINE ABOVE IS WHERE I AM DOUBTFUL ABOUT IF THERE IS ANY MISTAKE.PLEASE LET ME KNOW SIR.'''
    a=regr.coef_
    b=regr.intercept_
    y_temp=regr.predict(y_hat)
    y_temp=y_temp.reshape(-1,1)
    print(np.shape(y_hat))
    print(np.shape(y_temp))
    y_hat=np.append(y_hat,y_temp,axis=1)#Calculating y(i+1,t)=a+b*y(i,t) where a,b parameters are trained by the regressor
    print(np.shape(y_temp),np.shape(y_hat))
    MSE_test=mse(data_train,y_temp)#Calculate MSE
    tuples=(i,MSE_test,a,b)
    print (tuples)
    parameter_tuples.append(tuples)
    ALL_MSE.append(MSE_test)
    if MSE_min>MSE_test:
        break
total_iter=i
iteration=[i for i in range(total_iter)]
plt.text(x=29,y=177,s=('Iteration_End='+str(total_iter)),color='b')
plt.text(x=29,y=187,s=('Iteration_Threshold='+str(MSE_min)),color='b')
plt.plot(iteration,ALL_MSE[:total_iter],'r',marker='o')
plt.xlabel('iterations')
plt.ylabel('MSE')
plt.grid()
plt.show()
plt.text(x=1698,y=151,s=('MSE after AR-DAN2='+str(MSE_test)))
plt.text(x=1698,y=161,s=('MSE before AR-DAN2='+str(MSE_test_before)))
print ("COEFFICIENTS")
print('\n',regr.coef_)
plt.plot(datay[:220],data_train,'r',marker='o',label='original')#plotting the graph
plt.plot(datay[:220],y_temp,'b',marker='^',label='AR-DAN2')
plt.legend(loc='upper right')
plt.grid()
plt.show()
print(len(regr.coef_))


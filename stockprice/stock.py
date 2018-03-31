import numpy as np
import matplotlib.pyplot as plt
import glob
import csv
import operator
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense

path =r'2012'
allFiles = glob.glob(path + "/*.csv")
your_list=[]
dt=[]
cp=[]
dat=0
for f in allFiles:
    with open(f, 'r') as f1:
        reader = csv.reader(f1)
        dat=dat+1
       # print(dat)
        for o in reader:
            o[1]=dat
            your_list.append(o)
path =r'2013'
allFiles = glob.glob(path + "/*.csv")
for f in allFiles:
    with open(f, 'r') as f1:
        reader = csv.reader(f1)
        dat=dat+1
       # print(dat)
        for o in reader:
            o[1]=dat
            your_list.append(o)
path =r'2014' # use your path
allFiles = glob.glob(path + "/*.csv")
for f in allFiles:
    with open(f, 'r') as f1:
        reader = csv.reader(f1)
        dat=dat+1
       # print(dat)
        for o in reader:
            o[1]=dat            
            your_list.append(o)
path =r'2015' # use your path
allFiles = glob.glob(path + "/*.csv")
for f in allFiles:
    with open(f, 'r') as f1:
        reader = csv.reader(f1)
        dat=dat+1
      #  print(dat)
        for o in reader:
            o[1]=dat 
            your_list.append(o)
sorlist=sorted(your_list, key=operator.itemgetter(0,1), reverse=False)
na=np.array(sorlist)
#sorted(your_list,key=lambda x: x[3])

an=na[0][0]

lis1=[]
lis2=[]
lis3=[]
lis4=[]

for i in na:
    try:
        if(an!=i[0]):
            an=i[0]
            #print(i[0])
            
          #  model = LinearRegression()
          #  model.fit(dt[0:int(len(cp)/2)], cp[0:int(len(cp)/2)])
    
           # X_predict = dt
           # y_predict = model.predict(X_predict)
            
           # print(np.mean(np.abs((cp - y_predict) / cp)) * 100)
           # if((np.mean(np.abs((cp - y_predict) / cp)) * 100)<10):
            #    lis1.append([i[0],np.mean(np.abs((cp - y_predict) / cp)) * 100])
                
            #elif((np.mean(np.abs((cp - y_predict) / cp)) * 100)<25):
             #   lis2.append([i[0],np.mean(np.abs((cp - y_predict) / cp)) * 100])
           # elif((np.mean(np.abs((cp - y_predict) / cp)) * 100)<50):
            #    lis3.append([i[0],np.mean(np.abs((cp - y_predict) / cp)) * 100])
            #elif((np.mean(np.abs((cp - y_predict) / cp)) * 100)<100):
             #   lis4.append([i[0],np.mean(np.abs((cp - y_predict) / cp)) * 100])
           # plt.plot(dt,cp)
            #plt.show()
            dt=[]
            cp=[]
            
           # break
    
        cp.append(float(i[5]))
        dt.append([i[1]])
    except:
        print("excep")
        print(i)
   # print(i)
a=tuple(sorlist)
datetime=dt
#print(lis1)
#print(lis2)
#print(lis3)
#print(lis4)
trainx=[]
trainx1=[]
trainx2=[]
trainx3=[]
trainy=[]
cp=[]
dt=[]
k=0
kk=[]
dt1=[]
dt2=[]
dt3=[]
an=na[0][0]
kk.append(an)
for i in na:
    try:
        if(an!=i[0]):
            an=i[0]
            kk.append(an)
            trainx.append(dt)
            trainx1.append(dt1)
            trainx2.append(dt2)
            trainy.append(cp)
            trainx3.append(dt3)

            dt=[]
            cp=[]
            dt1=[]
            dt2=[]
            dt3=[]
            k=k+1
        
        cp.append(float(i[5]))
        dt.append(float(i[1]))
        dt1.append([float(i[1]),float(i[2]),float(i[3])])
        dt2.append([float(i[2])])
        dt3.append(float(i[1]))
    except:
        print("excep")
print("error")
for k in range(1,2):
    model = LinearRegression()
    model.fit(np.array(trainx[k][0:400]).reshape(-1,1),np.array(trainy[k][0:400]).reshape(-1,1))

    mm=model.predict(np.array(trainx[k]).reshape(-1,1))
    model.fit(trainx1[k][0:400],trainy[k][0:400])

    mmm=model.predict(np.array(trainx1[k]))

    clf = svm.SVR()
    clf.fit(trainx1[k][0:400], trainy[k][0:400])
    cc=clf.predict(trainx1[k])
    
 #   for l in range(1,30):
   #     print(kk[k],"Regration", mm[l],"ANN", mmm[l],"SVM",cc[l],"real",trainy[k][l])
    
    a=[]
    b=[]
    c=[]
    t=0;
    for i in mmm:
        if(mm[t]>mmm[t]):
            mmm[t]=mmm[t]+50*(mm[t]-mmm[t])/100
        else:
            mmm[t]=mmm[t]-50*(mmm[t]-mm[t])/100
        t=t+1
    
    # plotting t, a separately 
    model = Sequential()
    model.add(Dense(4, input_dim=3,kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(np.array(trainx1[k][0:400]),np.array(trainy[k][0:400]), epochs=150, batch_size=10)
    # evaluate the model
    predictions = model.predict(np.array(trainx1[k]))
    plt.plot(trainy[k], 'r')
    plt.plot(mm,'g')
    plt.plot(predictions, 'b')
    plt.plot(cc, 'y')
    
    plt.show()
    for l in range(1,30):
        print(kk[l], predictions[l],"asd")
    combined = np.vstack((np.array(trainx[k]).ravel(),np.array(trainy[k]).ravel(),np.array(mm).ravel(),np.array(predictions).ravel(),np.array(cc).ravel() )).T
    np.savetxt("test.csv", np.asarray(combined), delimiter=",")
    
#print(trainy[k][10])
#print(nn.predict([trainx[k][10]]))
#k=k+1
    #break
   
#print(np.mean(np.abs((cp - y_predict) / cp)) * 100)

#plt.plot(dt,cp)
#plt.show()
#print(na)

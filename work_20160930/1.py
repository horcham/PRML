import numpy as np
import pandas as pd
import os
import matplotlib

os.chdir('./trainingDigits')   #设置工作区间
#print(os.getcwd())
goal = [0,1,2,3,4,5,6,7,8,9]



train_data = []
for j in goal:
    filenum = 0
    while True:
        tem = []
        filename = str(goal[j]) + "_" + str(filenum) + '.txt'
        print("file:",filename)
        try:
            b = open(filename,'r')
        except FileNotFoundError:
            break

        c = b.read()
        b.close()
        d = c.replace("\n",'')
        size = len(d)
        i = 0
        for i in range(0,size-1):
            tem.append(int(d[i]))
        tem.append(1)  #偏置
        if j == 0:
            tem.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if j == 1:
            tem.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if j == 2:
            tem.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        if j == 3:
            tem.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        if j == 4:
            tem.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if j == 5:
            tem.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        if j == 6:
            tem.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if j == 7:
            tem.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        if j == 8:
            tem.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        if j == 9:
            tem.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


        train_data.append(tem)
        filenum = filenum + 1

train_data = np.array(train_data)
train_data_input = train_data[:,0:-10]
train_data_output = train_data[:,-10:]

W = np.linalg.pinv(np.matrix(train_data_input).T*np.matrix(train_data_input))*np.matrix(train_data_input).T*np.matrix(train_data_output)



os.chdir('../testDigits')   #设置工作区间
#print(os.getcwd())
goal = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
test_data = []
for j in goal:
    filenum = 0
    while True:
        tem = []
        filename = str(goal[j]) + "_" + str(filenum) + '.txt'
        print("file:",filename)
        try:
            b = open(filename,'r')
        except FileNotFoundError:
            break

        c = b.read()
        b.close()
        d = c.replace("\n",'')
        size = len(d)
        i = 0
        for i in range(0,size-1):
            tem.append(int(d[i]))
        tem.append(1)  #偏置
        if j == 0:
            tem.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if j == 1:
            tem.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if j == 2:
            tem.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        if j == 3:
            tem.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        if j == 4:
            tem.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if j == 5:
            tem.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        if j == 6:
            tem.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if j == 7:
            tem.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        if j == 8:
            tem.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        if j == 9:
            tem.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        test_data.append(tem)
        filenum = filenum + 1

test_data = np.array(test_data)
print("test_data shape:",test_data)

test_data_input = test_data[:,0:-10]
test_data_input_cal = W.T*test_data_input.T

test_data_output = test_data[:,-10:]
shape0 = test_data_output.shape
print("shape0[0]",shape0[0])


end0 = np.sum(abs((test_data_output.T-test_data_input_cal)/2))/(shape0[0])
print(end0)
print("end0:",end0)




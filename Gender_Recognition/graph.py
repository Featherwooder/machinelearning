# -*- coding: utf-8 -*-

from voicegender_optimized import main_
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import confusion_matrix
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'

accuracy_rate = []
times = 20
for i in range (times):
    accuracy_rate.append(main_())
a = np.arange(20)
x = np.array(accuracy_rate)
plt.plot(a,x[:,0],label=u'male')
plt.plot(a,x[:,1],label=u'female')
plt.title(u'Gender Recognition Accuracy Rate')
plt.xlabel(u'Time')
plt.ylabel(u'Accuracy')
plt.axis([0,20,0.84,1])
plt.legend(loc='best')
plt.show()
male_accu=x[:,0].mean()
female_accu=x[:,1].mean()
print('male average accuracy：%.2f%%   female average accuracy：%.2f%%'%(male_accu*100,female_accu*100))
print('male average mistake：%.2f%%   female average mistake：%.2f%%'%(100-male_accu*100,100-female_accu*100))

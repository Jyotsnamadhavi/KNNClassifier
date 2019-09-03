'''
Created on 03-Sep-2019

@author: Jyotsna Madhavi
'''
#Name the data sets
FRUITS_TRAIN = "kNN_weight_sphericity_train.csv"
FRUITS_TEST1 = "kNN_weight_sphericity_test.csv"

#Import pandas to read the csv
import pandas as pd
train = pd.read_csv(FRUITS_TRAIN)
print(train)

#Plot using mathplotlib
import matplotlib.pyplot as plt

apples=train[train['Label']=='A']
oranges=train[train['Label']=='O']

plt.plot(apples.Weight, apples.Sphericity, "ro")
plt.plot(oranges.Weight, oranges.Sphericity, "bo")
plt.xlabel("Weight")
plt.ylabel("Sphericity")
plt.legend(["Apples", "Oranges"])
plt.plot([300], [1], "ko")
plt.show()

#find the euclian distance to classify
import math
def distance(a, b):
    ''' a is the n-dimesnional co-ordinate of point 1
        b is the n-dimensional co-ordinate of point 2'''
    sqSum = 0
    for i in range(len(a)):
        sqSum += (a[i] - b[i]) ** 2
    return math.sqrt(sqSum)

#Compute KNN
def kNNClassifier(k, train, given):
    distances = []
    for t in train.values:              
        distances.append((distance(t[:2], given), t[2])) 
    distances.sort()            
    return distances[:k]

#print(kNNClassifier(3, train, (373, 1)))
#print(kNNClassifier(5, train, (373, 1)))
print(kNNClassifier(7,train, (250,1)))

#Count the max no for a a label in K values using counter
import collections
def kNNmax(k, train, given):
    tally = collections.Counter() #Initialize emoty counter
    for nn in kNNClassifier(k, train, given):
        tally.update(nn[-1])      #add each value to counter
        #print(tally)
    return tally.most_common(1)[0] #return the most common value among all 7 values

print(kNNmax(7, train, (250, 1))) #outputs ('A', 6)


testData = pd.read_csv(FRUITS_TEST1).values[:,:-1]
testResults = pd.read_csv(FRUITS_TEST1).values[:,-1]
results = []
for i, t in enumerate(testData):
    results.append(kNNmax(7, train, t)[0] == testResults[i])
print(results.count(True), "are matched with k=7")

results = []
for i, t in enumerate(testData):
    results.append(kNNmax(5, train, t)[0] == testResults[i])
print(results.count(True), "are matched with k=5")

results = []
for i, t in enumerate(testData):
    results.append(kNNmax(4, train, t)[0] == testResults[i])
print(results.count(True), "are matched with k=4")

results = []
for i, t in enumerate(testData):
    results.append(kNNmax(34, train, t)[0] == testResults[i])
print(results.count(True), "are matched with k=34")
#---------------
#Best K could be 4, 5,7











import numpy as np
from sklearn import svm
import csv


data = []

trainAmount = 0.1
numAtts = 30

classDict = {"M":0,"B":1}


with open("bcdata.csv",'rU') as csvfile:
	reader = csv.reader(csvfile,dialect=csv.excel_tab)
	use = False
	for row in reader:
		if use:
			splitRow = row[0].split(',')
			atts = np.array(splitRow[2:]).astype(float)
			label = splitRow[1]
			dataVal = np.append(atts,classDict[label])
			data.append(dataVal)
		else:
			use = True

data = np.array(data)

np.random.shuffle(data)

numData = len(data)

numTrainingData = int(numData*trainAmount)

fullDataAtts = data[:,:numAtts]
fullDataLabels = data[:,numAtts]

trainingData = data[:numTrainingData]
trainingAttributes = trainingData[:,:numAtts]
trainingLabels = trainingData[:,numAtts]

testData = data[numTrainingData:]
testAttributes = testData[:,:numAtts]
testLabels = testData[:,numAtts]


possC = [0.01,0.05,0.1,0.17,0.25]
possK = ['linear']

bestC = -1.0
bestK = ''
bestAcc = 0.0

for c in possC :
	for k in possK:		
		clf = svm.SVC(C=c,kernel=k)
		clf.fit(np.array(trainingAttributes),np.array(trainingLabels))
		acc = clf.score(np.array(testAttributes),np.array(testLabels))
		if(acc>bestAcc):
			bestAcc=acc
			bestC=c
			bestK=k

print(bestC)
print(bestK)
clf = svm.SVC(C=bestC,kernel=bestK)
clf.fit(np.array(trainingAttributes),np.array(trainingLabels))
acc = clf.score(np.array(testAttributes),np.array(testLabels))
print(acc)







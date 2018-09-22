import numpy as np
import csv

USE_OLD = False
SAVE_THIS = True

data = []
numAtts = 30

trainAmount = 0.75

classDict = {"M":0,"B":1}

if USE_OLD:
	data = np.loadtxt("scrambledData.txt")
else:

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

	if(SAVE_THIS):
		np.savetxt("scrambledData.txt",data)

numData = len(data)

numTrainingData = int(numData*trainAmount)

trainingData = data[:numTrainingData]

testData = data[numTrainingData:]
numTestData = len(testData)


STOP_ENTROPY = 0.1

classes = 2

nodes = []

if(USE_OLD):
	nodes = np.loadtxt("DecisionTreeNodes.txt")

maxLevels = 6

def entropy(c):
	classCounts = np.zeros(classes)
	for i in range(len(c)):
		classIndex = int(c[i][numAtts])
		classCounts[classIndex]+=1
	classProbs = np.divide(classCounts,float(len(c)))

	totalEntropy = 0.0

	for i in range(classes):
		p = classProbs[i]
		if(p>0):
			entropy = -p*np.log(p)
			totalEntropy += entropy

	return totalEntropy

def informationGain(c,c1,c2):
	previousEntropy = entropy(c)

	newEntropy = weightedEntropy(c1,c2)

	return previousEntropy-newEntropy

def weightedEntropy(c1,c2):

	combLength = float(len(c1)+len(c2))

	c1Entropy = entropy(c1)
	c2Entropy = entropy(c2)
	wC1 = len(c1)/combLength
	wC2 = len(c2)/combLength
	return wC1*c1Entropy+wC2*c2Entropy

def splitSorted(c,p):
	minEntropy = entropy(c)
	b1f = []
	b2f = []
	for i in p:
		c1 = c[:i]
		c2 = c[i:]
		e = weightedEntropy(c1,c2)
		if e<minEntropy:
			b1f = c1
			b2f = c2
			minEntropy = e
	return [b1f,b2f,entropy(c)-minEntropy]

def bestSplit(d):

	b1 = []
	b2 = []

	maxIG = 0

	attUsed = -1

	for i in range(numAtts):
		sortedData = d[d[:,i].argsort()]
		possible = []
		for j in range(len(d)-1):
			first = sortedData[j][i]
			next = sortedData[j+1][i]
			if not first == next:
				possible.append(j+1)
		[f1,f2,iG] = splitSorted(sortedData,possible)

		if(iG>maxIG):
			b1 = f1
			b2 = f2
			maxIG = iG
			attUsed = i
			thresh = 0.5*(f1[len(f1)-1][i]+f2[0][i])

	return [attUsed,thresh,b1,b2,maxIG]

def probDist(c):
	classCounts = np.zeros(classes)
	for i in range(len(c)):
		classIndex = int(c[i][numAtts])
		classCounts[classIndex]+=1

	return np.divide(classCounts,float(len(c)))

def classCounts(c):
	classCounts = np.zeros(classes)
	for i in range(len(c)):
		classIndex = int(c[i][numAtts])
		classCounts[classIndex]+=1

	return classCounts

def nextNode():
	ID = len(nodes)
	nodes.append(np.zeros(5))
	return ID

def tree(d,nodeID,maxIterations):
	e = entropy(d)
	print("NODE %i"%(nodeID))
	cDist = classCounts(d)
	print("NODE DIST")
	print(cDist)
	print("ENTROPY: %f"%(e))
	print("")
	if(e<STOP_ENTROPY or maxIterations==0):
		print("LEAF")
		
		prediction = np.argmax(cDist)
		nodes[nodeID] = np.array([1,prediction,0,0,0])

		print("_______________________")
	else:
		print("SPLITTING...")
		[attribute,threshold,part1,part2,infoGain] = bestSplit(d)
		print("Attribute %i, Threshold: %f"%(attribute,threshold))
		print("INFORMATION GAIN: %f"%(infoGain))
		p1ID = nextNode()
		p2ID = nextNode()
		nodes[nodeID] = np.array([0,attribute,threshold,p1ID,p2ID])
		print("SPLIT INTO %i AND %i"%(p1ID,p2ID))
		tree(part1,p1ID,maxIterations-1)
		tree(part2,p2ID,maxIterations-1)


if not USE_OLD :
	root = nextNode()
	print("Beginning TREE...")
	tree(trainingData,root,maxLevels)
	if(SAVE_THIS):
		np.savetxt("DecisionTreeNodes.txt",nodes)

def predict(d):
	nodeID = 0
	while(True):
		if(nodes[nodeID][0]==1):
			return nodes[nodeID][1]
		else:
			att = int(nodes[nodeID][1])
			threshold = nodes[nodeID][2]
			dAtt = d[att]
			if(d[att]<threshold):
				nodeID = int(nodes[nodeID][3])
			else:
				nodeID = int(nodes[nodeID][4])

def accuracy(d):

	correct = 0

	for i in range(len(d)):
		dataVal = d[i]
		prediction = predict(dataVal)
		correctLabel = dataVal[numAtts]
		if(prediction==correctLabel):
			correct +=1

	return float(correct)/len(data)

print(accuracy(trainingData))






















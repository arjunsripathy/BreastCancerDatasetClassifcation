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

STOP_ENTROPY = 0.05

classes = 3
randomNodes = []
maxLevels = 5
rAttsAvailable = 2
numTrees = 25
uProp = float(rAttsAvailable)/numAtts

PRINT_TREE_LOG = False

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
	minEntropy = entropy(c)+1.0
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

def bestRandomSplit(d):

	b1 = []
	b2 = []

	maxIG = -1.0

	attUsed = -1
	thresh = -1

	split = True

	usableAtts = []
	for i in range(numAtts):
		if((rAttsAvailable-len(usableAtts))==(numAtts-i)):
			usableAtts.append(i)
		elif(np.random.normal()<uProp):
			usableAtts.append(i)

		if(len(usableAtts)==rAttsAvailable):
				break

	for i in usableAtts:
		sortedData = d[d[:,i].argsort()]
		possible = []
		for j in range(len(d)-1):
			first = sortedData[j][i]
			next = sortedData[j+1][i]
			if not first == next:
				possible.append(j+1)

		if(len(possible)==0):
			split = False
			break

		[f1,f2,iG] = splitSorted(sortedData,possible)
		
		if(iG>maxIG):
			b1 = f1
			b2 = f2
			maxIG = iG
			attUsed = i

			firstHalfMax = f1[len(f1)-1]
			firstHalfMax = firstHalfMax[i]
			secondHalfMin = f2[0][i]
			thresh = 0.5*(firstHalfMax+secondHalfMin)


	return [attUsed,thresh,b1,b2,maxIG,split]

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

def nextRandomNode(treeID):
	if(len(randomNodes)==treeID):
		randomNodes.append([])
	ID = len(randomNodes[treeID])
	randomNodes[treeID].append(np.zeros(5))
	return ID

def randomTree(d,treeID,nodeID,maxIterations):


	e = entropy(d)
	cDist = classCounts(d)

	if(PRINT_TREE_LOG):
		print("NODE %i"%(nodeID))
		print("NODE DIST")
		print(cDist)
		print("ENTROPY: %f"%(e))
		print("")

	if(e<STOP_ENTROPY or maxIterations==0):
		
		if(PRINT_TREE_LOG):
			print("LEAF")
			print("_______________________")

		prediction = np.argmax(cDist)
		randomNodes[treeID][nodeID] = np.array([1,prediction,e,len(d),0])
	else:
		if(PRINT_TREE_LOG):
			print("SPLITTING...")

		[attribute,threshold,part1,part2,infoGain,split] = bestRandomSplit(d)

		if(split):

			if(PRINT_TREE_LOG):
				print("Attribute %i, Threshold: %f"%(attribute,threshold))
				print("INFORMATION GAIN: %f"%(infoGain))

			p1ID = nextRandomNode(treeID)
			p2ID = nextRandomNode(treeID)
			randomNodes[treeID][nodeID] = np.array([0,attribute,threshold,p1ID,p2ID])

			if(PRINT_TREE_LOG):
				print("SPLIT INTO %i AND %i"%(p1ID,p2ID))

			randomTree(part1,treeID,p1ID,maxIterations-1)
			randomTree(part2,treeID,p2ID,maxIterations-1)
		else:
			if(PRINT_TREE_LOG):
				print("LEAF")
				print("_______________________")

			prediction = np.argmax(cDist)
			randomNodes[treeID][nodeID] = np.array([1,prediction,e,len(d),0])


def weightedLeafEntropy(treeID,nD):
	wle = 0.0
	for i in range(len(randomNodes[treeID])):
		node = randomNodes[treeID][i]
		if(node[0]==1):
			e = node[2]
			q = node[3]
			wle += e*q
	wle /= nD
	return wle


def createForest(d,nT):
	for i in range(nT):

		treeID = i
		root = nextRandomNode(treeID)

		nD = len(d)
		baggingIndexes = np.random.randint(0,nD,[nD])
		baggedData = []

		for j in range(nD):
			index = np.random.randint(0,nD)
			baggedData.append(data[baggingIndexes[index]])

		baggedData = np.array(baggedData)

		print("Beginning TREE %i ..."%(treeID))
		randomTree(baggedData,treeID,root,maxLevels)
		print("COMPLETED!")

def treePredict(d,treeID):
	tree = randomNodes[treeID]
	cNode = tree[0]
	while True:
		if(cNode[0]==1):
			return int(cNode[1])
		att = int(cNode[1])
		thresh = cNode[2]
		thisAtt = d[att]
		if(thisAtt<thresh):
			cNode = tree[int(cNode[3])]
		else:
			cNode = tree[int(cNode[4])]

def forestPredict(d):
	treeVotes = np.zeros(classes)
	for i in range(numTrees):
		treeVotes[treePredict(d,i)]+=1
	return np.argmax(treeVotes)

def accuracy(d):
	correct = 0
	for i in range(len(d)):
		dataVal = d[i]
		dClass = dataVal[numAtts]
		pClass = forestPredict(dataVal)
		if(dClass==pClass):
			correct+=1
	return float(correct)/len(d)

createForest(trainingData,numTrees)
print("Training Accuracy: %f"%(accuracy(trainingData)))
print("Test Accuracy: %f"%(accuracy(testData)))



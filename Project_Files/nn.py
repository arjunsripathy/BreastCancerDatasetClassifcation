import numpy as np
import csv
import tensorflow as tf


data = []
labels = []
numAtts = 30

numClasses = 2

classDict = {"M":0,"B":1}
orderedData = []

with open("bcdata.csv",'rU') as csvfile:
		reader = csv.reader(csvfile,dialect=csv.excel_tab)

		use = False
		for row in reader:
			if use:
				orderedData.append(row)
			else:
				use = True

np.random.shuffle(orderedData)


for row in orderedData:
	splitRow = row[0].split(',')
	atts = np.array(splitRow[2:]).astype(float)
	label = classDict[splitRow[1]]
	oneHotLabel = np.zeros(numClasses)
	oneHotLabel[label] = 1
	data.append(atts)
	labels.append(oneHotLabel)	

numData = len(data)

trainAmount = 0.75

numTrainingData = int(trainAmount*numData)

trainingData = data[:numTrainingData]
trainingLabels = labels[:numTrainingData]


testData = data[numTrainingData:]
testLabels = labels[numTrainingData:]
numTestData = len(testData)


def weight_variable(shape,wStdDev):
  initial = tf.truncated_normal(shape, stddev=wStdDev)
  return tf.Variable(initial)

def bias_variable(shape,init):
  initial = tf.constant(init, shape=shape)
  return tf.Variable(initial)


numHidden = 50
wSD = 0.1
bI = 1.0
LR = 1e-3
MBSIZE = 10
EPOCHS = 16

x = tf.placeholder(tf.float32,[None,numAtts])
y_ = tf.placeholder(tf.float32,[None,numClasses])


Wxh = weight_variable([numAtts,numHidden],wSD)
biasH = bias_variable([numHidden],bI)

hidden = tf.nn.relu(tf.matmul(x,Wxh)+biasH)


Why = weight_variable([numHidden,numClasses],wSD)

y = tf.matmul(hidden,Why)

crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
train_step = tf.train.AdamOptimizer(learning_rate = LR).minimize(crossEntropy)
correctPrediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction,tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(EPOCHS):
		for j in range(int(numTrainingData/MBSIZE)):
			dataBatch = trainingData[j*(MBSIZE):(j+1)*MBSIZE]
			labelBatch = trainingLabels[j*(MBSIZE):(j+1)*MBSIZE]
			train_step.run(feed_dict={x:dataBatch,y_:labelBatch})

		[cost,acc] = sess.run([crossEntropy,accuracy],feed_dict={x:trainingData,y_:trainingLabels})
		print("Epoch %i, Cost: %f, Accuracy: %f"%(i,cost,acc))
		[cost,acc] = sess.run([crossEntropy,accuracy],feed_dict={x:testData,y_:testLabels})
		print("Test Cost: %f, Test Accuracy: %f"%(cost,acc))











import parser
import bagofwords
import math
import string
import numpy as np

class multinomial:
    alpha=0
    class_log_prior_=[]
    feature_log_prob_=[]
    def __init__(self,alpha=1.0):
        self.alpha=alpha
    def fit(self, trainingData, y):
        count_sample = trainingData.shape[0]
        separated = [[x for x, t in zip(trainingData, y) if t == c] for c in np.unique(y)]
        for i in separated:
            self.class_log_prior_.append(np.log(float(len(i)) / float(count_sample)))
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
        return self

    def predict_log_proba(self, Data):
        return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_
                for x in Data]

    def predict(self, Data):
        return np.argmax(self.predict_log_proba(Data), axis=1)

    def accuracy(self, classData,predictions):
        i=0
        count=0.0
        for prediction in predictions:
            if prediction==classData[i]:
                count+=1
            i+=1
        return count/len(predictions)

def main(posTrainFile,negTrainFile,TestFile1,TestFile2):    
    bag=bagofwords.BagOfWords(posTrainFile,negTrainFile) 
    trainingData=np.array(bag.vectors)
    classData=np.array(bag.classVectors)

    test1Reviews=parser.process(TestFile1)
    test1Freqs=[]
    for review in test1Reviews:
        parser.freq_extraction(review,test1Freqs)
    test1Vecs=np.array(parser.genVectors(test1Freqs,bag.vocab))
    temp=[0.0]*len(test1Reviews)
    posClassData=np.array(temp)
  
    test2Reviews=parser.process(TestFile2)
    test2Freqs=[]
    for review in test2Reviews:
        parser.freq_extraction(review,test2Freqs)
    test2Vecs=np.array(parser.genVectors(test2Freqs,bag.vocab))
    temp=[1.0]*len(test2Reviews)
    negClassData=np.array(temp)
  

    nb = multinomial().fit(trainingData, classData)
    test1Predictions=nb.predict(test1Vecs.sum(axis=1))
    print("Test File 1 (positive) accuracy: {}%").format(nb.accuracy(posClassData,test1Predictions)*100)

    test2Predictions=nb.predict(test2Vecs.sum(axis=1))
    print("Test File 2 (negative) accuracy: {}%").format(nb.accuracy(negClassData,test2Predictions)*100)


main("training_pos.txt","training_neg.txt","test_pos_public.txt","test_neg_public.txt")
import pickle
import os
import string
import parser

class BagOfWords:
    vocab=[]
    vectors=[]
    classVectors=[]
    
    def __init__(self,posReviewsFile,negReviewsFile):
        pickle_filepathNeg = "./negWords.pickle"
        pickle_filepathPos = "./posWords.pickle"
        negReviewFreqs=[]
        posReviewFreqs=[]
        negWords=[]
        posWords=[]
        posReviews=parser.process(posReviewsFile)
        negReviews=parser.process(negReviewsFile)
        for review in posReviews:
            self.word_extraction(review,posWords,posReviewFreqs)
        for review in negReviews:
            self.word_extraction(review,negWords,negReviewFreqs)
 
        if not os.path.exists(pickle_filepathNeg):
            with open(pickle_filepathNeg, 'w') as pickle_handle:
                pickle.dump(negWords, pickle_handle)
            with open(pickle_filepathPos, 'w') as pickle_handle:
                pickle.dump(posWords, pickle_handle)  
                
        else:
            with open(pickle_filepathNeg,"r") as pickle_handle:
                negWords = pickle.load(pickle_handle)
            with open(pickle_filepathPos,"r") as pickle_handle:
                posWords = pickle.load(pickle_handle)
      
        self.vocab=sorted(list(set(posWords+negWords)))
        self.genVectors(posReviewFreqs,negReviewFreqs,posWords,negWords)
        
    
    def word_extraction(self,review,wordslist,reviewFreqList):
        reviewDict={}
        for word in review.split():
            if word.translate(None, string.punctuation).lower() not in wordslist:
                wordslist.append(word.translate(None, string.punctuation).lower())
            if word.translate(None, string.punctuation).lower() not in reviewDict.keys():
                reviewDict[word.translate(None, string.punctuation).lower()]=1
            else:
                reviewDict[word.translate(None, string.punctuation).lower()]+=1
        reviewFreqList.append(reviewDict)

    def genEmptyVector(self,posWords,negWords):
        vector=[0.0]*(len(posWords)+len(negWords))
        return vector

    def genVectors(self,posReviewFreqs,negReviewFreqs,posWords,negWords):
        for reviewFreq in posReviewFreqs:
            vector=self.genEmptyVector(posWords,negWords)
            for word,freq in reviewFreq.items():
                if(word in self.vocab):
                    vector[self.vocab.index(word)]=freq
            self.vectors.append(vector)
        for reviewFreq in negReviewFreqs:
            vector=self.genEmptyVector(posWords,negWords)
            for word,freq in reviewFreq.items():
                if(word in self.vocab):
                    vector[self.vocab.index(word)]=freq
            
            self.vectors.append(vector)
        temp=[0.0]*len(posReviewFreqs)
        temp2=[1.0]*len(negReviewFreqs)
        self.classVectors=temp+temp2
            
        

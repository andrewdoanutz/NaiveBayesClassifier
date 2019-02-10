import string

def process(dataset):
    reviews=[]
    with open(dataset,'r') as f:
        for line in f:
            for word in line.split("<br /><br />"):
                reviews.append(word)
    return reviews


def freq_extraction(review,reviewFreqList):
        reviewDict={}
        for word in review.split():
            if word.translate(None, string.punctuation).lower() not in reviewDict.keys():
                reviewDict[word.translate(None, string.punctuation).lower()]=1
            else:
                reviewDict[word.translate(None, string.punctuation).lower()]+=1
        reviewFreqList.append(reviewDict)

def genEmptyVector(vocab):
        vector=[0.0]*(len(vocab))
        return vector

def genVectors(ReviewFreqs,vocab):
        vectors=[]
        for reviewFreq in ReviewFreqs:
            vector=genEmptyVector(vocab)
            for word,freq in reviewFreq.items():
                if(word in vocab):
                    vector[vocab.index(word)]=freq
            vectors.append(vector)
        return vectors
        



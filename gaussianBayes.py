import bagofwords
import tfidf
import math
import parser
import string
import pickle #caching
import os

class GN:
    
    posProbs={}
    negProbs={}
    stopWords=["a", "about", "above", "across", "after", "afterwards", 
    "again", "all", "almost", "alone", "along", "already", "also","although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "as", "at", "be", "became", "because", "become","becomes", "becoming", "been", "before", "behind", "being", "beside", "besides", "between", "beyond", "both", "but", "by","can", "cannot", "cant", "could", "couldnt", "de", "describe", "do", "done", "each", "eg", "either", "else", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "find","for","found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next","no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part","perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should","since", "sincere","so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take","than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they",
    "this", "those", "though", "through", "throughout","thru", "thus", "to", "together", "too", "toward", "towards","under", "until", "up", "upon", "us",
    "very", "was", "we", "well", "were", "what", "whatever", "when","whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "who", "whoever", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"]+ ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
       
    def __init__(self,option,train1,train2):
        if option=="bag":
            pickle_filepathBPProbs = "./bpProbs.pickle"
            pickle_filepathBNProbs = "./bnProbs.pickle"
            print(option)
            self.ex=bagofwords.BagOfWords(train1,train2)
            if not os.path.exists(pickle_filepathBPProbs):
                print("training data")
                self.fit(self.ex.posVectors,self.ex.negVectors,self.ex.posVocab,self.ex.negVocab)
                with open(pickle_filepathBPProbs, 'wb') as pickle_handle:
                    pickle.dump(self.posProbs, pickle_handle)
                with open(pickle_filepathBNProbs, 'wb') as pickle_handle:
                    pickle.dump(self.negProbs, pickle_handle)
                print("data cached")
            else:
                print("loading training data")
                with open(pickle_filepathBPProbs, 'rb') as pickle_handle:
                    self.posProbs=pickle.load(pickle_handle)
                with open(pickle_filepathBNProbs, 'rb') as pickle_handle:
                    self.negProbs=pickle.load(pickle_handle)
                print("data loaded")

        elif option=="tf":
            pickle_filepathTPProbs = "./tpProbs.pickle"
            pickle_filepathTNProbs = "./tnProbs.pickle"
            print(option)
            self.ex=tfidf.tfidf(train1,train2)
            if not os.path.exists(pickle_filepathTPProbs):
                print("training data")
                self.fit(self.ex.posVectors,self.ex.negVectors,self.ex.posVocab,self.ex.negVocab)
                with open(pickle_filepathTPProbs, 'wb') as pickle_handle:
                    pickle.dump(self.posProbs, pickle_handle)
                with open(pickle_filepathTNProbs, 'wb') as pickle_handle:
                    pickle.dump(self.negProbs, pickle_handle)
                print("data cached")
            else:
                print("loading training data")
                with open(pickle_filepathTPProbs, 'rb') as pickle_handle:
                    self.posProbs=pickle.load(pickle_handle)
                with open(pickle_filepathTNProbs, 'rb') as pickle_handle:
                    self.negProbs=pickle.load(pickle_handle)
                print("data loaded")
        
        

    def fit(self,posVectors,negVectors,posVocab,negVocab):
        posMean={}
        negMean={}
        posVar={}
        negVar={}
        numPosRevs=len(posVectors)
        numNegRevs=len(negVectors)
        for word in posVocab:
            posFreq=0.0
            for vector in posVectors:
                if word in vector.keys():
                    posFreq+=vector[word]
            posMean[word]=(posFreq/numPosRevs)
            temp=0.0
            for vector in posVectors:
                value=0.0
                if word in vector.keys():
                    value=vector[word]
                temp+=(value-posMean[word])**2
            posVar[word]=temp/numPosRevs
            if word in self.posProbs.keys():
                continue
            else:
                pProb=0.0
                for vector in posVectors:
                    prevP=0.1 #fixes domain error with estimate of prob
                    value=0.0
                    if word in vector.keys():
                        value=vector[word]
                    prevP=self.gaussianProb(value,posMean[word],posVar[word],prevP)
                    pProb+=prevP
                self.posProbs[word]=pProb
        for word in negVocab:
            negFreq=0.0
            for vector in negVectors:
                if word in vector.keys():
                    negFreq+=vector[word]
            negMean[word]=(negFreq/numNegRevs)
            temp=0.0
            for vector in negVectors:
                value=0.0
                if word in vector.keys():
                    value=vector[word]
                temp+=(value-negMean[word])**2
            negVar[word]=temp/numPosRevs
            if word in self.negProbs.keys():
                continue
            else:
                nProb=0.0
                for vector in negVectors:
                    prevN=0.1 #fixes domain error with estimate of prob
                    value=0.0
                    if word in vector.keys():
                        value=vector[word]
                    prevN=self.gaussianProb(value,negMean[word],negVar[word],prevN)
                    nProb+=prevN
                self.negProbs[word]=nProb

    def gaussianProb(self,wordFreq,wordMean,wordVar,prev):
        right=0
        exponentnum=(wordFreq-wordMean)*(wordFreq-wordMean)
        exponentden=2*wordVar
        right+=(-1*exponentnum/exponentden)
        left=math.log(1/math.sqrt(2*math.pi*(wordVar)))
        #if(right*left)==0.0:
            #return prev
        return (left*right)

    def predict(self,inputFile1, inputFile2):
        print("predicting")
        reviews1=parser.process(inputFile1)
        posReviews1=0.0
        negReviews1=0.0
        for review in reviews1:
            posWords1=0.0
            negWords1=0.0
            for word in review.split():
                posProb1=0.0
                negProb1=0.0
                cleanedWord=word.translate(str.maketrans('','',string.punctuation)).lower()
                if cleanedWord in self.stopWords:
                    continue
                else:
                    if cleanedWord in self.ex.posVocab:
                        posProb1+=self.posProbs[cleanedWord]
                    if cleanedWord in self.ex.negVocab:
                        negProb1+=self.negProbs[cleanedWord]
                if(posProb1>negProb1):
                    posWords1+=1
                else:
                    negWords1+=1
            if(posWords1>negWords1):
                posReviews1+=1
            else:
                negReviews1+=1
        if(posReviews1>negReviews1):
            print("Test File 1 (positive) accuracy: {} %".format(posReviews1/(posReviews1+negReviews1)*100))
        else:
            print("Test File 1 (negative) accuracy: {} %".format(negReviews1/(posReviews1+negReviews1)*100))
        reviews2=parser.process(inputFile2)
        posReviews2=0.0
        negReviews2=0.0
        for review in reviews2:
            posWords2=0.0
            negWords2=0.0
            for word in review.split():
                posProb2=0.0
                negProb2=0.0
                cleanedWord=word.translate(str.maketrans('','',string.punctuation)).lower()
                if cleanedWord in self.stopWords:
                    continue
                else:
                    if cleanedWord in self.ex.posVocab:
                        posProb2+=self.posProbs[cleanedWord]
                    if cleanedWord in self.ex.negVocab:
                        negProb2+=self.negProbs[cleanedWord]
                if(posProb2>negProb2):
                    posWords2+=1
                else:
                    negWords2+=1
            if(posWords2>negWords2):
                posReviews2+=1
            else:
                negReviews2+=1
        if(posReviews2>negReviews2):
            print("Test File 2 (positive) accuracy: {} %".format(posReviews2/(posReviews2+negReviews2)*100))
        else:
            print("Test File 2 (negative) accuracy: {} %".format(negReviews2/(posReviews2+negReviews2)*100))
gn=GN("bag","training_pos.txt","training_neg.txt")
print("predicting")
gn.predict("testinput.txt","testinput2.txt")
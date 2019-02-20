import bagofwords
import tfidf
import math
import parser
import string
import pickle #caching
import os

class GN:
    posMean={}
    negMean={}
    posVar={}
    negVar={}
    stopWords=["a", "about", "above", "across", "after", "afterwards", 
    "again", "all", "almost", "alone", "along", "already", "also","although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "as", "at", "be", "became", "because", "become","becomes", "becoming", "been", "before", "behind", "being", "beside", "besides", "between", "beyond", "both", "but", "by","can", "cannot", "cant", "could", "couldnt", "de", "describe", "do", "done", "each", "eg", "either", "else", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "find","for","found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next","no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part","perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should","since", "sincere","so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take","than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they",
    "this", "those", "though", "through", "throughout","thru", "thus", "to", "together", "too", "toward", "towards","under", "until", "up", "upon", "us",
    "very", "was", "we", "well", "were", "what", "whatever", "when","whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "who", "whoever", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"]+ ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
       
    def __init__(self,option,train1,train2):
        if option=="bag":
            pickle_filepathBPMean = "./bpMeanpickle"
            pickle_filepathBNMean = "./bnMeanpickle"
            pickle_filepathBNVar = "./bnVar.pickle"
            pickle_filepathBPVar = "./bpVar.pickle"
            print(option)
            self.ex=bagofwords.BagOfWords(train1,train2)
            if not os.path.exists(pickle_filepathBPMean):
                print("training gaussian data")
                self.fit(self.ex.posVectors,self.ex.negVectors,self.ex.posVocab,self.ex.negVocab)
                with open(pickle_filepathBPMean, 'wb') as pickle_handle:
                    pickle.dump(self.posMean, pickle_handle)
                with open(pickle_filepathBNMean, 'wb') as pickle_handle:
                    pickle.dump(self.negMean, pickle_handle)
                with open(pickle_filepathBNVar, 'wb') as pickle_handle:
                    pickle.dump(self.negVar, pickle_handle)
                with open(pickle_filepathBPVar, 'wb') as pickle_handle:
                    pickle.dump(self.posVar, pickle_handle)
                print("gaussian data cached")
            else:
                print("loading gaussian training data")
                with open(pickle_filepathBPMean, 'rb') as pickle_handle:
                    self.posMean=pickle.load(pickle_handle)
                with open(pickle_filepathBNMean, 'rb') as pickle_handle:
                    self.negMean=pickle.load(pickle_handle)
                with open(pickle_filepathBPVar, 'rb') as pickle_handle:
                    self.posVar=pickle.load(pickle_handle)
                with open(pickle_filepathBNVar, 'rb') as pickle_handle:
                    self.negVar=pickle.load(pickle_handle)
                print("data gaussian loaded")

        elif option=="tf":
            pickle_filepathTPMean = "./tpMeanpickle"
            pickle_filepathTNMean = "./tnMeanpickle"
            pickle_filepathTNVar = "./tnVar.pickle"
            pickle_filepathTPVar = "./tpVar.pickle"
            print(option)
            self.ex=tfidf.tfidf(train1,train2)
            if not os.path.exists(pickle_filepathTPMean):
                print("training gaussian data")
                self.fit(self.ex.posVectors,self.ex.negVectors,self.ex.posVocab,self.ex.negVocab)
                with open(pickle_filepathTPMean, 'wb') as pickle_handle:
                    pickle.dump(self.posMean, pickle_handle)
                with open(pickle_filepathTNMean, 'wb') as pickle_handle:
                    pickle.dump(self.negMean, pickle_handle)
                with open(pickle_filepathTNVar, 'wb') as pickle_handle:
                    pickle.dump(self.negVar, pickle_handle)
                with open(pickle_filepathTPVar, 'wb') as pickle_handle:
                    pickle.dump(self.posVar, pickle_handle)
                print("gaussian data cached")
            else:
                print("loading gaussian training data")
                with open(pickle_filepathTPMean, 'rb') as pickle_handle:
                    self.posMean=pickle.load(pickle_handle)
                with open(pickle_filepathTNMean, 'rb') as pickle_handle:
                    self.negMean=pickle.load(pickle_handle)
                with open(pickle_filepathTPVar, 'rb') as pickle_handle:
                    self.posVar=pickle.load(pickle_handle)
                with open(pickle_filepathTNVar, 'rb') as pickle_handle:
                    self.negVar=pickle.load(pickle_handle)
                print("gaussian data loaded")
        
        

    def fit(self,posVectors,negVectors,posVocab,negVocab):
        
        numPosRevs=len(posVectors)
        numNegRevs=len(negVectors)
        for word in posVocab:
            posFreq=0.0
            for vector in posVectors:
                if word in vector.keys():
                    posFreq+=vector[word]
            self.posMean[word]=(posFreq/numPosRevs)
            temp=0.0
            for vector in posVectors:
                value=0.0
                if word in vector.keys():
                    value=vector[word]
                temp+=(value-self.posMean[word])**2
            self.posVar[word]=temp/numPosRevs
            
        for word in negVocab:
            negFreq=0.0
            for vector in negVectors:
                if word in vector.keys():
                    negFreq+=vector[word]
            self.negMean[word]=(negFreq/numNegRevs)
            temp=0.0
            for vector in negVectors:
                value=0.0
                if word in vector.keys():
                    value=vector[word]
                temp+=(value-self.negMean[word])**2
            self.negVar[word]=temp/numPosRevs
            

    def gaussianProb(self,wordFreq,wordMean,wordVar):
        right=0
        exponentnum=(wordFreq-wordMean)*(wordFreq-wordMean)
        exponentden=2*wordVar
        right+=(-1*exponentnum/exponentden)
        left=math.log(1/math.sqrt(2*math.pi*(wordVar)))
        return (right+left)

    def predict(self,inputFile1, inputFile2,option):
        if option=="bag":
            print("predicting")
            reviews1=parser.process(inputFile1)
            posReviews1=0.0
            negReviews1=0.0
            revFreqs1=[]
            for review in reviews1: #find frequencies in test file
                reviewDict={}
                for word in review.split():
                    cleanedWord=word.translate(str.maketrans('','',string.punctuation)).lower()
                    if cleanedWord not in reviewDict.keys():
                        reviewDict[cleanedWord]=1
                    else:
                        reviewDict[cleanedWord]+=1
                revFreqs1.append(reviewDict)
            i1=0
            for review in reviews1:
                posWords1=0.0
                negWords1=0.0
                for word in review.split():
                    cleanedWord=word.translate(str.maketrans('','',string.punctuation)).lower()
                    if cleanedWord in self.stopWords:
                        continue
                    else:
                        if cleanedWord in self.ex.posVocab:
                            freq=0.0
                            if cleanedWord in revFreqs1[i1].keys():
                                freq+=revFreqs1[i1][cleanedWord]
                            posWords1+=self.gaussianProb(freq,self.posMean[cleanedWord],self.posVar[cleanedWord])
                        if cleanedWord in self.ex.negVocab:
                            freq=0.0
                            if cleanedWord in revFreqs1[i1].keys():
                                freq+=revFreqs1[i1][cleanedWord]
                            negWords1+=self.gaussianProb(freq,self.negMean[cleanedWord],self.negVar[cleanedWord])
                i1+=1
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
            revFreqs2=[]
            for review in reviews2: #find frequencies in test file
                reviewDict={}
                for word in review.split():
                    cleanedWord=word.translate(str.maketrans('','',string.punctuation)).lower()
                    if cleanedWord not in reviewDict.keys():
                        reviewDict[cleanedWord]=1
                    else:
                        reviewDict[cleanedWord]+=1
                revFreqs2.append(reviewDict)
            i2=0
            for review in reviews2:
                posWords2=0.0
                negWords2=0.0
                for word in review.split():
                    cleanedWord=word.translate(str.maketrans('','',string.punctuation)).lower()
                    if cleanedWord in self.stopWords:
                        continue
                    else:
                        if cleanedWord in self.ex.posVocab:
                            freq=0.0
                            if cleanedWord in revFreqs2[i2].keys():
                                freq+=revFreqs2[i2][cleanedWord]
                            posWords2+=self.gaussianProb(freq,self.posMean[cleanedWord],self.posVar[cleanedWord])
                        if cleanedWord in self.ex.negVocab:
                            freq=0.0
                            if cleanedWord in revFreqs2[i2].keys():
                                freq+=revFreqs2[i2][cleanedWord]
                            negWords2+=self.gaussianProb(freq,self.negMean[cleanedWord],self.negVar[cleanedWord])
                i2+=1
                if(posWords2>negWords2):
                    posReviews2+=1
                else:
                    negReviews2+=1
            if(posReviews2>negReviews2):
                print("Test File 2 (positive) accuracy: {} %".format(posReviews2/(posReviews2+negReviews2)*100))
            else:
                print("Test File 2 (negative) accuracy: {} %".format(negReviews2/(posReviews2+negReviews2)*100))
        elif option=="tf":
            print("predicting")
            reviews1=parser.process(inputFile1)
            posReviews1=0.0
            negReviews1=0.0
            idfs1={}
            revFreqs1=[]
            for review in reviews1: #find frequencies in test file
                reviewDict={}
                for word in review.split():
                    cleanedWord=word.translate(str.maketrans('','',string.punctuation)).lower()
                    if cleanedWord not in reviewDict.keys():
                        reviewDict[cleanedWord]=1
                    else:
                        reviewDict[cleanedWord]+=1
                revFreqs1.append(reviewDict)
            for review in reviews1: #find idf scores
                for word in review.split():
                    cleanedWord=word.translate(str.maketrans('','',string.punctuation)).lower()
                    revsWordIn=0.0
                    for review in revFreqs1:
                        if cleanedWord in review.keys():
                            revsWordIn+=1
                    idfs1[cleanedWord]=math.log(len(reviews1)/revsWordIn)
            i1=0
            for review in reviews1:
                posWords1=0.0
                negWords1=0.0
                for word in review.split():
                    cleanedWord=word.translate(str.maketrans('','',string.punctuation)).lower()
                    if cleanedWord in self.stopWords:
                        continue
                    else:
                        if cleanedWord in self.ex.posVocab:
                            tfscore=0.0
                            #if cleanedWord in revFreqs1[i1].keys():
                            tfscore+=revFreqs1[i1][cleanedWord]/sum(revFreqs1[i1].values())
                            posWords1+=self.gaussianProb(tfscore*idfs1[cleanedWord],self.posMean[cleanedWord],self.posVar[cleanedWord])
                        if cleanedWord in self.ex.negVocab:
                            tfscore=0.0
                            #if cleanedWord in revFreqs1[i1].keys():
                            tfscore+=revFreqs1[i1][cleanedWord]/sum(revFreqs1[i1].values())
                            negWords1+=self.gaussianProb(tfscore*idfs1[cleanedWord],self.negMean[cleanedWord],self.negVar[cleanedWord])
                if(posWords1>negWords1):
                    posReviews1+=1
                else:
                    negReviews1+=1
                i1+=1
            if(posReviews1>negReviews1):
                print("Test File 1 (positive) accuracy: {} %".format(posReviews1/(posReviews1+negReviews1)*100))
            else:
                print("Test File 1 (negative) accuracy: {} %".format(negReviews1/(posReviews1+negReviews1)*100))
            reviews2=parser.process(inputFile2)
            posReviews2=0.0
            negReviews2=0.0
            revFreqs2=[]
            idfs2={}
            for review in reviews2: #find frequencies in test file
                reviewDict={}
                for word in review.split():
                    cleanedWord=word.translate(str.maketrans('','',string.punctuation)).lower()
                    if cleanedWord not in reviewDict.keys():
                        reviewDict[cleanedWord]=1
                    else:
                        reviewDict[cleanedWord]+=1
                revFreqs2.append(reviewDict)

            for review in reviews2: #find idf scores
                for word in review.split():
                    cleanedWord=word.translate(str.maketrans('','',string.punctuation)).lower()
                    revsWordIn=0.0
                    for review in revFreqs2:
                        if cleanedWord in review.keys():
                            revsWordIn+=1
                    idfs2[cleanedWord]=math.log(len(reviews2)/revsWordIn)
            i2=0
            for review in reviews2:
                posWords2=0.0
                negWords2=0.0
                for word in review.split():
                    cleanedWord=word.translate(str.maketrans('','',string.punctuation)).lower()
                    if cleanedWord in self.stopWords:
                        continue
                    else:
                        if cleanedWord in self.ex.posVocab:
                            tfscore=0.0
                            #if cleanedWord in revFreqs2[i2].keys():
                            tfscore+=revFreqs2[i2][cleanedWord]/sum(revFreqs2[i2].values())
                            posWords2+=self.gaussianProb(tfscore*idfs2[cleanedWord],self.posMean[cleanedWord],self.posVar[cleanedWord])
                        if cleanedWord in self.ex.negVocab:
                            tfscore=0.0
                            #if cleanedWord in revFreqs2[i2].keys():
                            tfscore+=revFreqs2[i2][cleanedWord]/sum(revFreqs2[i2].values())
                            negWords2+=self.gaussianProb(tfscore*idfs2[cleanedWord],self.negMean[cleanedWord],self.negVar[cleanedWord])
                if(posWords2>negWords2):
                    posReviews2+=1
                else:
                    negReviews2+=1
                i2+=1
            if(posReviews2>negReviews2):
                print("Test File 2 (positive) accuracy: {} %".format(posReviews2/(posReviews2+negReviews2)*100))
            else:
                print("Test File 2 (negative) accuracy: {} %".format(negReviews2/(posReviews2+negReviews2)*100))


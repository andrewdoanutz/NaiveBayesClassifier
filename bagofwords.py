import pickle #caching
import os
import string
import parser

class BagOfWords:
    posVocab=[]
    negVocab=[]
    posVectors=[]
    negVectors=[]
    stopWords=["a", "about", "above", "across", "after", "afterwards", 
    "again", "all", "almost", "alone", "along", "already", "also","although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "as", "at", "be", "became", "because", "become","becomes", "becoming", "been", "before", "behind", "being", "beside", "besides", "between", "beyond", "both", "but", "by","can", "cannot", "cant", "could", "couldnt", "de", "describe", "do", "done", "each", "eg", "either", "else", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "find","for","found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next","no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part","perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should","since", "sincere","so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take","than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they",
    "this", "those", "though", "through", "throughout","thru", "thus", "to", "together", "too", "toward", "towards","under", "until", "up", "upon", "us",
    "very", "was", "we", "well", "were", "what", "whatever", "when","whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "who", "whoever", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"]+ ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        
    def __init__(self,posReviewsFile,negReviewsFile):
        pickle_filepathPVec = "./PVec.pickle"
        pickle_filepathNVec = "./NVec.pickle"
        pickle_filepathNVocab = "./NVocab.pickle"
        pickle_filepathPVocab = "./PVocab.pickle"
        
        if not os.path.exists(pickle_filepathPVec):
            posReviews=parser.process(posReviewsFile)
            negReviews=parser.process(negReviewsFile)
            print("extracting words")
            for review in posReviews:
                self.word_extraction(review,self.posVocab,self.posVectors)
            for review in negReviews:
                self.word_extraction(review,self.negVocab,self.negVectors)
            with open(pickle_filepathNVec, 'wb') as pickle_handle:
                pickle.dump(self.negVectors, pickle_handle)
            with open(pickle_filepathPVec, 'wb') as pickle_handle:
                pickle.dump(self.posVectors, pickle_handle)
            with open(pickle_filepathPVocab, 'wb') as pickle_handle:
                pickle.dump(self.posVocab, pickle_handle)
            with open(pickle_filepathNVocab, 'wb') as pickle_handle:
                pickle.dump(self.negVocab, pickle_handle)
            print("bag data cached")
        else:
            print("loading bag training data")
            with open(pickle_filepathNVec, 'rb') as pickle_handle:
                self.negVectors = pickle.load(pickle_handle)
            with open(pickle_filepathPVec, 'rb') as pickle_handle:
                self.posVectors = pickle.load(pickle_handle)
            with open(pickle_filepathPVocab, 'rb') as pickle_handle:
                self.posVocab = pickle.load(pickle_handle)
            with open(pickle_filepathNVocab, 'rb') as pickle_handle:
                self.negVocab = pickle.load(pickle_handle)
            print("bag data loaded")
        
    
    
    def word_extraction(self,review,wordslist,reviewFreqList):
        reviewDict={}
        for word in review.split():
            cleanedWord=word.translate(str.maketrans('','',string.punctuation)).lower()
            if cleanedWord in self.stopWords:
                continue
            if cleanedWord not in wordslist:
                wordslist.append(word.translate(str.maketrans('','',string.punctuation)).lower())
            if cleanedWord not in reviewDict.keys():
                reviewDict[cleanedWord]=1
            else:
                reviewDict[cleanedWord]+=1
        reviewFreqList.append(reviewDict)


        

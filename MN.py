import MNbag
import parser
import math
import string
class MN:
    stopWords= ["a", "about", "above", "across", "after", "afterwards", 
"again", "all", "almost", "alone", "along", "already", "also","although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "as", "at", "be", "became", "because", "become","becomes", "becoming", "been", "before", "behind", "being", "beside", "besides", "between", "beyond", "both", "but", "by","can", "cannot", "cant", "could", "couldnt", "de", "describe", "do", "done", "each", "eg", "either", "else", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "find","for","found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next","no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part","perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should","since", "sincere","so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take","than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they",
"this", "those", "though", "through", "throughout","thru", "thus", "to", "together", "too", "toward", "towards","under", "until", "up", "upon", "us",
"very", "was", "we", "well", "were", "what", "whatever", "when","whence", "whenever", "where", "whereafter", "whereas", "whereby",
"wherein", "whereupon", "wherever", "whether", "which", "while", "who", "whoever", "whom", "whose", "why", "will", "with",
"within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"]+ ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    def __init__(self,train1,train2):
        self.bag=MNbag.MNbag(train1,train2)
        
    def predict_log(self, word, wordslist, vocabNum, totalWords):
        wordFreq=0.0
        if word in wordslist.keys():
            wordFreq=wordslist[word]
        return math.log((1.0+wordFreq)/(totalWords+vocabNum))
        
        
    def predict(self,inputFile1, inputFile2):
        print("predicting")
        reviews1=parser.process(inputFile1)
        reviews2=parser.process(inputFile2)
        posLen=len(self.bag.posVocab.keys())
        negLen=len(self.bag.negVocab.keys())
        posWordsTotal=sum(self.bag.posVocab.values())
        negWordsTotal=sum(self.bag.negVocab.values())
        posReviews1=0.0
        negReviews1=0.0
        for review in reviews1:
            posProb1=0.0
            negProb1=0.0
            for word in review.split():
                cleanedWord=word.translate(str.maketrans('','',string.punctuation)).lower()
                if cleanedWord in self.stopWords:
                    continue
                else:
                    posProb1+=self.predict_log(cleanedWord,self.bag.posVocab,posLen,posWordsTotal)
                    negProb1+=self.predict_log(cleanedWord,self.bag.negVocab,negLen,negWordsTotal)
            if(posProb1>negProb1):
                posReviews1+=1
            else:
                negReviews1+=1
        if(posReviews1>negReviews1):
            print("Test File 1 (positive) accuracy: {} %".format(posReviews1/(posReviews1+negReviews1)*100))
        else:
            print("Test File 1 (negative) accuracy: {} %".format(negReviews1/(posReviews1+negReviews1)*100))
        posReviews2=0.0
        negReviews2=0.0
        for review in reviews2:
            posProb2=0.0
            negProb2=0.0
            for word in review.split():
                cleanedWord=word.translate(str.maketrans('','',string.punctuation)).lower()
                if cleanedWord in self.stopWords:
                    continue
                else:
                    posProb2+=self.predict_log(cleanedWord,self.bag.posVocab,posLen,posWordsTotal)
                    negProb2+=self.predict_log(cleanedWord,self.bag.negVocab,negLen,negWordsTotal)
            if(posProb2>negProb2):
                posReviews2+=1
            else:
                negReviews2+=1
        if(posReviews2>negReviews2):
            print("Test File 2 (positive) accuracy: {} %".format(posReviews2/(posReviews2+negReviews2)*100))
        else:
            print("Test File 2 (negative) accuracy: {} %".format(negReviews2/(posReviews2+negReviews2)*100))


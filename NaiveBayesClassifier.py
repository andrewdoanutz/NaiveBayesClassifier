import gaussianBayes
import MN
import sys

print("multinomial with bag of words")
print("-"*20)
MN=MN.MN(sys.argv[1],sys.argv[2])
MN.predict(sys.argv[3],sys.argv[4])
print("gaussian with bag of words")
print("-"*20)
bagGN=gaussianBayes.GN("bag",sys.argv[1],sys.argv[2])
bagGN.predict(sys.argv[3],sys.argv[4])
print("gaussian with tfidf")
print("-"*20)
tfGN=gaussianBayes.GN("tf",sys.argv[1],sys.argv[2])
tfGN.predict(sys.argv[3],sys.argv[4])

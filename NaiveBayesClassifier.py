import gaussianBayes
import multi
import sys

print("\nmultinomial with bag of words")
print("-"*20)
bagMN=multi.MN(sys.argv[1],sys.argv[2])
bagMN.predict(sys.argv[3],sys.argv[4])

print("\ngaussian with bag of words")
print("-"*20)
bagGN=gaussianBayes.GN("bag",sys.argv[1],sys.argv[2])
bagGN.predict(sys.argv[3],sys.argv[4],"bag")


print("\ngaussian with tfidf")
print("-"*20)
tfGN=gaussianBayes.GN("tf",sys.argv[1],sys.argv[2])
tfGN.predict(sys.argv[3],sys.argv[4],"tf")

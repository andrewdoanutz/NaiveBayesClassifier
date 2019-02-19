import string

def process(dataset):
    reviews=[]
    with open(dataset,'r') as f:
        for line in f:
            for word in line.split("<br /><br />"):
                reviews.append(word)
    return reviews


       
        



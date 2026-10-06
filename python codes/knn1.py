import math
import random

data=[]
file=open("iris_dataset.csv", "r")
lines=file.readlines()
file.close()
for line in lines[1:]:
    row=line.strip().split(",")
    for i in range(4):
        row[i]=float(row[i])
    data.append(row)

    # Shuffle the dataset
random.seed(42)
random.shuffle(data)

train=data[:120]
test=data[120:]

k=3
correct=0
def distance(row1, row2):
    s=0
    for i in range(4):
        s+=(row1[i] - row2[i]) ** 2
    return math.sqrt(s)
for test_row in test:
    distances=[]
    for train_row in train:
        d=distance(test_row, train_row)
        distances.append([d, train_row[4]])
    distances.sort()
    votes={}
    for i in range(k):
        label=distances[i][1]
        if label in votes:
            votes[label]+=1
        else:
            votes[label]=1
    prediction=max(votes, key=votes.get)
    print("Actual:", test_row[4], "Predicted:", prediction)
    if prediction==test_row[4]:
        correct += 1
accuracy=(correct / len(test)) * 100
print("\nCorrect Predictions:", correct)
print("Accuracy=", accuracy, "%")
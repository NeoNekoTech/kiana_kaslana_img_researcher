import matplotlib.pyplot as plt 
import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
stats_file = os.path.join(current_dir, "training_stats.json")

with open(stats_file, "r") as file:
    data = json.loads(file.read())

trainloss = data["train_loss"]
val_loss = data["val_loss"]
epoch = data["epochs"]
acc_train = data["train_acc"]
acc_test = data["val_acc"]

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.plot(epoch,trainloss)
plt.title("trainloss")

plt.subplot(2,2,2)
plt.plot([5,10,15,20,25,30],val_loss)
plt.title("val loss")

plt.subplot(2,2,3)
plt.plot(epoch,acc_train)
plt.title("train acc")

plt.subplot(2,2,4)
plt.plot([5,10,15,20,25,30],acc_test)
plt.title("val acc")


plt.show()



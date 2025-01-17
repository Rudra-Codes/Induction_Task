"""
Made by Rudra In Spyder

Used Pytorch framework

First of all preprocessed the data with help of provided code by you which includes :

1) Prepared a panda library containg image details 
2) x_train tensor containg image transformed to tensor
    
    Note - I used pillow library to load image(new Library sikhi hai dekh rhe ho)

Then Comes model, I used a single linear layer as model beacuse
CNN se accuracy kam arahi thi I don't know why 

what do you mean by maine aise hi CNN sikha aur implement kiya tha (crying inside)

And also a hidden baat - (top secret) - I converted image to grayscale(channel = 1) to reduce my
computation task as i thought colour would not matter so much in this particular classification

As a result I can locally run this in less than 1 minute in my pc
and still got 99.37 acc in Data 1 (flex pro max) 
"""
# %%
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

dimension = [512,512]

def createdataframe(dir, dimension, test=False):
    if test == False:
        image_name = []
        image_paths = []
        labels = []
        x_train = torch.empty([1,(dimension[0]*dimension[1])])
        # print(x_train.shape)
        
        for label in os.listdir(dir):
            for imagename in os.listdir(os.path.join(dir, label)):
                image_name.append(imagename)
                path = os.path.join(dir, label, imagename)
                image_paths.append(path)
                labels.append(label)
                x_train = torch.cat((x_train,image_processing(path, dimension)))
            print(label, "completed")
        return image_name, image_paths, labels, x_train[1:,:].to(dtype=torch.float32)
    
    elif test == True:
        image_name = []
        image_paths = []
        
        x_test = torch.empty([1,(dimension[0]*dimension[1])])
        # print(x_train.shape)
        
        
        for imagename in os.listdir(dir):
            image_name.append(imagename)
            path = os.path.join(dir, imagename)
            image_paths.append(path)
            
            x_test = torch.cat((x_test,image_processing(path, dimension)))
        print("Test Data Loaded")
        return image_name, image_paths, x_test[1:,:].to(dtype=torch.float32)
        


def image_processing(str_path,dimension):
  transformation = transforms.Compose([(transforms.Resize((dimension[0],dimension[1]))),(transforms.ToTensor())])  
  image = Image.open(str_path).convert("L")
  image_tensor = transformation(image).view([1,(dimension[0]*dimension[1])])
  return image_tensor

# Directories - 
# For Competiton - 1
TRAIN_DIR = "D:\Induction_Task\Data\Train"  
TEST_DIR = "D:\Induction_Task\Data\Test"

# For Competiton - 2
# TRAIN_DIR = "D:\Induction_Task\Data\Train_Images"
# TEST_DIR = "D:\Induction_Task\Data\Test_Images"

# Creating x_train
train = pd.DataFrame()
train['image_name'], train['image_path'], train['label'], x_train = createdataframe(TRAIN_DIR,dimension,test=False)

# Creating x_test
test = pd.DataFrame()
test['image_name'], test['image_path'], x_test = createdataframe(TEST_DIR,dimension,test=True)

# Creating tensor of labels - 
train['label_encoded'] = train.label.map({'AI':0,'Real':1})
y_train = torch.tensor(train.label_encoded,dtype=torch.float32).view([len(train),1])

# %%
# Model 
class Model(torch.nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

n_features = x_train.shape[1]
model = Model(n_features)

# 2) Loss
num_epochs = 600
learning_rate = 0.001
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    
    loss.backward()
    optimizer.step()

    
    optimizer.zero_grad()

    if (epoch+1) % 100 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
        
print(f'Final Loss = {loss.item():.4f}')

# Creating Csv File
y_answer = model(x_test)

test_samples = x_test.shape[0]

answer = pd.DataFrame(columns=['Id','Label'])
answer.Id = test.image_name.str.split('.').str[0]
answer.Label = y_answer.detach().view([test_samples]).numpy().round()

answer.Label = answer.Label.map({0:'AI', 1:'Real'})

answer.to_csv('submission_2.csv', index=False)

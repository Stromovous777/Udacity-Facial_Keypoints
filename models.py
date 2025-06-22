## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

'''
class Net(nn.Module):

    def __init__(self, drop_p=0.5):
        super(Net, self).__init__()


        #############################################
        # CNN --> convergence problems and also with overfitting... :-(
        #############################################
        
        #ConvLayer output size = (Input size - Kernel size + 2 * Padding) / Stride + 1
        
        # (224-3+2*0)/1 + 1 = 222 => After pool = 111
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0, bias=False)
        
        # (111-3+2*0)/1 + 1 = 109 => After pool = 54
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0, bias=False)

        # (54-3+2*0)/1 + 1 = 52 => After pool = 26
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)

        # (26-3+2*0)/1 + 1 = 24 => After pool = 12
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        
        # (12-3+2*0)/1 + 1 = 10 => After pool = 5
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer
        self.fc1 = nn.Linear(128*5*5, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.drop = nn.Dropout(drop_p)
        self.fc3 = nn.Linear(512, 136)
        

        
    def forward(self, x):
        
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))
        x = F.relu(self.pool(self.conv5(x)))
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        #print("Dimension: ", x.size(1) )
        
        # two linear layers with dropout in between
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x) 
                
        # a modified x, having gone through all the layers of your model, should be returned
        return x  
'''


class Net(nn.Module):

    def __init__(self, drop_p=0.5):
        super(Net, self).__init__()

        #############################################
        # CNN: with batch normalization -> everthing good :-)
        #############################################
        
        #ConvLayer output size = (Input size - Kernel size + 2 * Padding) / Stride + 1
        
        # (224-3+2*0)/1 + 1 = 222 => After pool = 111
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(num_features=8,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # (111-3+2*0)/1 + 1 = 109 => After pool = 54
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(num_features=16,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # (54-3+2*0)/1 + 1 = 52 => After pool = 26
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.batchNorm3 = nn.BatchNorm2d(num_features=32,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # (26-3+2*0)/1 + 1 = 24 => After pool = 12
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.batchNorm4 = nn.BatchNorm2d(num_features=64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # (12-3+2*0)/1 + 1 = 10 => After pool = 5
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.batchNorm5 = nn.BatchNorm2d(num_features=128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer
        self.fc1 = nn.Linear(128*5*5, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.drop = nn.Dropout(drop_p)
        self.fc3 = nn.Linear(512, 136)
        

        
    def forward(self, x):
        
        x = F.relu(self.pool(self.batchNorm1(self.conv1(x))))
        x = F.relu(self.pool(self.batchNorm2(self.conv2(x))))
        x = F.relu(self.pool(self.batchNorm3(self.conv3(x))))
        x = F.relu(self.pool(self.batchNorm4(self.conv4(x))))
        x = F.relu(self.pool(self.batchNorm5(self.conv5(x))))
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        #print("Dimension: ", x.size(1) )
        
        # two linear layers with dropout in between
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x) 

        
        #x = self.fc1(x)
        #x = F.relu(x)
        #x = self.drop(x)
        #x = self.fc2(x) 
        #x = F.relu(x)
        #x = self.drop(x)
        #x = self.fc3(x) 
                
        # a modified x, having gone through all the layers of your model, should be returned
        return x  


    


######################################################################################################################
    '''
    ....Backup....
    
    def __init__(self, drop_p=0.5):
        super(Net, self).__init__()
        
        #############################################
        # CNN: with batch normalization 
        #############################################
        
        #ConvLayer output size = (Input size - Kernel size + 2 * Padding) / Stride + 1
        
        # (224-3+2)/2 + 1 = 224 => After pool = 112
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(num_features=8,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # (112-3+2)/2 + 1 = 112 => After pool = 56
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(num_features=16,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        
        # (56-3+2)/2 + 1 = 526 => After pool = 28
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(num_features=32,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # (28-3+2)/2 + 1 = 28 => After pool = 14
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(num_features=32,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        
        # (14-3+2)/2 + 1 = 14 => After pool = 7
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.batchNorm5 = nn.BatchNorm2d(num_features=64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Fully connected layer
        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.drop = nn.Dropout(drop_p)
        self.fc3 = nn.Linear(512, 136)
        

        
    def forward(self, x):
        
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
        x = F.relu(self.batchNorm5(self.conv5(x)))
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        #print("Dimension: ", x.size(1) )
        
        # two linear layers with dropout in between
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x) 
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc3(x) 
                
        # a modified x, having gone through all the layers of your model, should be returned
        return x  
  '''
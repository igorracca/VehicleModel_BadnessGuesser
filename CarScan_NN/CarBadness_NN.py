import datetime
import os
import numpy as np
import scipy.io as sio
import torch
from torch import nn, optim
from tqdm import tqdm
from tqdm import trange
import matplotlib.pyplot as plt

device = torch.device("cuda:0")

def read_data(base_path='./Data',split_size=[0.6, 0.2, 0.2]):  #  the split_size gives how much of the data goes to the train/test.
    """
    Reads all of the .mat files in the given base_path, and returns a dict with the data found there.
    :param split_size:
    :param base_path: The directory that should be read in.
    :return: a dict, containing the EES and difference tensors.
    """
    i = 0
    for file in os.listdir(base_path):
        i = i + 1
    pbar = tqdm(total=i)

    data_dict = {}
    for file in os.listdir(base_path):
        num, data_type = file.split('_')
        data_type = data_type.split('.')[0]
        num = int(num)
        if "EES" in data_type:
            tensor_in = sio.loadmat(os.path.join(base_path, file))['EES_value']
            tensor_in = torch.FloatTensor(tensor_in).squeeze(0)
        else:
            tensor_in = sio.loadmat(os.path.join(base_path, file))['Kulonbseg']
            tensor_in = torch.FloatTensor(tensor_in)
        try:
            data_dict[num][data_type] = tensor_in
        except KeyError:
            data_dict[num] = {data_type: tensor_in}
        pbar.update()
    pbar.close()

    new_data = []
    for key in data_dict.keys():
        new_data.append(data_dict[key])
    np.random.shuffle(new_data)
    if isinstance(split_size, list):
        training_samples = int(split_size[0] * len(new_data))
        valid_samples = int(split_size[1] * len(new_data))
        test_samples = int(split_size[2] * len(new_data))
        while sum([training_samples, valid_samples, test_samples]) != len(new_data):
            training_samples += 1
        #split samples
        new_datadict = {'train': new_data[ : training_samples],
                        'validation': new_data[training_samples : training_samples + valid_samples],
                        'test': new_data[-test_samples : ]}
        
    else:
        new_datadict = {'train': new_data,
                        'validation': new_data,
                        'test': new_data}
    print("Adatbetöltés kész")
    return new_datadict



class Classificator:

    #-------------------------
    # Label   Class
    #-------------------------
    # 0:      Not damaged
    # 1:      Slightly damaged
    # 2:      Damaged
    # 3:      Very damaged
    #-------------------------
    
    def getClassName(self, c):
        if c==0:
            n = "Not Damaged"
        elif c==1:
            n = "Slightly damaged"
        elif c==2:
            n = "Damaged"
        elif c==3:
            n = "Very damaged"
        return n;
    
    def getClass(self, x):
        x = int(x)
        
        if x==0:
            return 0      
        if x<=100/3:
            return 1
        elif  x>100/3 and x<200/3:
            return 2
        else:
            return 3
    
    # Compare Prediction and Label classes
    def isCorrect(self, prediction, label):
        return self.getClass(prediction) == self.getClass(label)

class CarBadnessGuesser(nn.Module):
    def __init__(self, lr=0.01):
        super(CarBadnessGuesser, self).__init__()

        self.valid_freq = 10

        self.model = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=3, kernel_size=(10, 5, 5), stride=(10, 5, 5)),
            nn.BatchNorm3d(3),
            nn.Conv3d(in_channels=3, out_channels=2, kernel_size=5),
            nn.BatchNorm3d(2),
            nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3),
            nn.BatchNorm3d(1),
            nn.AdaptiveMaxPool3d((1, 1, 10)),
        ) 
        
        self.linear = nn.Sequential(
            nn.Linear(in_features=10, out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5, out_features=1),
            nn.ReLU()
        )
        if torch.cuda.is_available():
            self.linear.cuda()
            self.model.cuda()

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.linear.parameters()), lr=lr)
        
        self.classificator = Classificator()

    def forward(self, x):
        conv_out = self.model(x.unsqueeze(0).unsqueeze(0))
        return self.linear(conv_out.squeeze(-2).squeeze(-2))

    def train(self, epochs=50):
        b_loss = []
        v_loss = []
    
        for epoch in trange(epochs):
            t_correct = 0
            total = 0
            for step, data in enumerate(dataset["train"]):
                input_data = data['KUL'].cuda()
                label = data['EES'].cuda()
                
                prediction = self(input_data)
                loss = self.loss_fn(prediction, label)
                #print('training- ', 'y^', prediction.item(), 'y', label.item(), loss.item())
                
                loss.backward()     
                self.optimizer.step()
                b_loss.append(loss.item())
                
                #check if its prediction matches label class
                if( self.classificator.isCorrect(prediction.item(), label.item()) ):
                    t_correct += 1;
                total += 1;
                
                self.zero_grad()
            if epoch % self.valid_freq and epoch != 0:
                #calculate the training accuracy
                t_acc = t_correct/total * 100
                #print(f'Batch loss: {loss.item()}', flush=True)
             
                vloss, v_acc = self.validation()
                v_loss.append(vloss)
                print(f'Validation loss:', vloss, flush=True)

                
        print('Training acc:', t_acc, '%')
        print('Validation acc: ', v_acc, '%')  
        plt.plot(v_loss)
        plt.ylabel('Validation loss')
        plt.show()
        plt.plot(b_loss)
        plt.ylabel('Batch loss')
        plt.show()

    def validation(self):
        """
        Runs the validation phase of the training
        :return: The validation loss average
        """
        v_correct = 0
        total = 0
        average_loss = 0
        step = 0
        
        for step, data in enumerate(dataset['validation']):
            with torch.no_grad():
                input_data = data['KUL'].cuda()
                label = data['EES'].cuda()
                
                prediction = self(input_data)
                loss = self.loss_fn(prediction, label)               
                #print('validation- ', 'y^', prediction.item(), 'y', label.item(), loss.item())
                #print('v_correct', v_correct, 'total',total)
                
                average_loss += loss.item()
                
                #check if its correct
                c = self.classificator.isCorrect(prediction.item(), label.item())
                if(c == True):
                    v_correct += 1;
                total += 1;
        
        #calculate the validation accuracy
        v_acc = v_correct/total * 100
        return average_loss / (step + 1), v_acc
        print("Validation is complete")
    
    def test(self):
        """
        Runs the evaluation of the network.
        :return: average loss for the test
        """
        t_correct = 0
        total = 0
        average_loss = 0
        step = 0
        
        for step, data in enumerate(dataset['test']):
            with torch.no_grad():
                input_data = data['KUL'].cuda()
                prediction = self(input_data)
                
                loss = self.loss_fn(prediction, data['EES'].cuda())
                
                print('---------------------------------')
                print()
                print('Prediction: ', prediction.item())
                print('Excpected:  ', data['EES'].cuda().item())   
                print()
                prediction_label = self.classificator.getClass(prediction.item())
                print('Class:          ', self.classificator.getClassName(prediction_label))
                expected_label = self.classificator.getClass(data['EES'].cuda())
                print('Expected Class: ', self.classificator.getClassName(expected_label))
                print()
                print('loss: ', loss.item())
                print()
                
                #check if its correct
                c = self.classificator.isCorrect(prediction.item(), data['EES'].cuda().item())
                if(c == True):
                    t_correct += 1;
                total += 1;
                
                average_loss += loss.item()

        #calculate the validation accuracy
        t_acc = t_correct/total * 100          
        average_loss = average_loss / step   
        print()
        print('---------------------------------')
        print('Test Accuracy: ', t_acc, ' %')
        print('Average Loss:  ', average_loss)
        print('---------------------------------')
        print()
        print("Test is completed")       
        return average_loss

    def save_weights(self, save_dir="./training"):
        """
        Saves weights to the given directory plus the timestamp
        :return: none
        """
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%d")
        save_path = os.path.join(save_dir, timestamp)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save model within Training folder with timestamp together with preivous models
        torch.save(self.state_dict(), os.path.join(save_path, 'model.weights'))
        # save last model within Weights folder (overwrite)
        torch.save(self.state_dict(), './weights/model.weights')

    def load_weights(self):
        self.load_state_dict = torch.load('./weights/model.weights')
        print("loading weights is complete")

if __name__ == "__main__":
    
    # set mode
    TRAIN = False
    TEST = True

    # Load the data and get the splitted dataset
    dataset = read_data()

    # instanciate the NN
    net = CarBadnessGuesser()
    torch.backends.cudnn.enabled = False

    # Train, Validate and Save the model
    if(TRAIN):
        net.train()
        net.save_weights()
    
    # Load the model and Test
    if(TEST):
        net.load_weights()
        net.test()
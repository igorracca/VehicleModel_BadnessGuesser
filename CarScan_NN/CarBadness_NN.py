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
        new_datadict = {'train': new_data[:training_samples],
                        'validation': new_data[training_samples + 1:training_samples + valid_samples],
                        'test': new_data[-test_samples:]}
    else:
        new_datadict = {'train': new_data,
                        'validation': new_data,
                        'test': new_data}
    print("Adatbetöltés kész")
    return new_datadict

    class CarBadnessGuesser(nn.Module):
    def __init__(self, lr=0.01):
        super(CarBadnessGuesser, self).__init__()

        self.dataset = read_data()
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


    # 0: no damage, 1: slightly damaged, 2: damaged, 3: very damaged
    def getClass(self, x):
        if x==0:
            return 0      
        x = int(x)
        if x<=100/3:
            return 1
        elif  x>100/3 and x<200/3:
            return 2
        else:
            return 3
    
    # compare prediction and label classes
    def isCorrect(self, prediction, label):
        return self.getClass(prediction) == self.getClass(label)

    def forward(self, x):
        conv_out = self.model(x.unsqueeze(0).unsqueeze(0))
        return self.linear(conv_out.squeeze(-2).squeeze(-2))

    def train(self, epochs=50):
        b_loss = []
        v_loss = []
    
        for epoch in trange(epochs):
            t_correct = 0
            total = 1
            for step, data in enumerate(self.dataset["train"]):
                input_data = data['KUL'].cuda()
                label = data['EES'].cuda()
                
                prediction = self(input_data)
                loss = self.loss_fn(prediction, label)
                #print('training- ', 'y^', prediction.item(), 'y', label.item(), loss.item())
                #print('t_correct', t_correct, 'total',total)
                
                loss.backward()     
                self.optimizer.step()
                b_loss.append(loss.item())
                
                #check if its prediction matches label class
                if( self.isCorrect(prediction.item(), label.item()) ):
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
        self.save_weights()
        
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
        total = 1
        average_loss = 0
        step = 0
        
        for step, data in enumerate(self.dataset['validation']):
            with torch.no_grad():
                input_data = data['KUL'].cuda()
                label = data['EES'].cuda()
                
                prediction = self(input_data)
                loss = self.loss_fn(prediction, label)               
                #print('validation- ', 'y^', prediction.item(), 'y', label.item(), loss.item())
                #print('v_correct', v_correct, 'total',total)
                
                average_loss += loss.item()
                
                #check if its correct
                c = self.isCorrect(prediction.item(), label.item())
                if(c == True):
                    v_correct += 1;
                total += 1;
        
        #calculate the validation accuracy
        v_correct = v_correct/total * 100  
        return average_loss / (step + 1), v_correct
        print("Validation is complete")

    def test(self):
        """
        Runs the evaluation of the network.
        :return: average loss for the test
        """
        average_loss = 0
        step = 0
        for step, data in enumerate(self.dataset['test']):
            with torch.no_grad():
                input_data = data['KUL'].cuda()
                prediction = self(input_data)
                loss = self.loss_fn(prediction, data['EES'].cuda())
                average_loss += loss.item()
        return average_loss / step
        print("the test is complete")

    def save_weights(self, save_dir="./training"):
        """
        Saves weights to the given directory plus the timestamp
        :return: none
        """
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%d")
        save_path = os.path.join(save_dir, timestamp)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(list(self.model.parameters()) +
                   list(self.linear.parameters()), os.path.join(save_path, 'model.weights'))
        print("saving weights is complete")

    def load_weights(self):
        self.optimizer = torch.load('model.weights')
        print("loading is complete")

if __name__ == "__main__":
    learner = CarBadnessGuesser()
    torch.backends.cudnn.enabled = False
    learner.train()
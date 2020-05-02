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

def read_data(base_path='./Data',split_size=[0.8, 0.1, 0.1]):  #  the split_size gives how much of the data goes to the train/test.
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
    if isinstance(split_size, tuple):
        training_samples = int(split_size[0] * len(new_data))
        valid_samples = int(split_size[1] * len(new_data))
        test_samples = int(split_size[2] * len(new_data))
        while sum([training_samples, valid_samples, test_samples]) != len(new_data):
            training_samples += 1

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
    def __init__(self, lr=0.0001):
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

    def forward(self, x):
        conv_out = self.model(x.unsqueeze(0).unsqueeze(0))
        return self.linear(conv_out.squeeze(-2).squeeze(-2))

    def train(self, epochs=50):
        b_loss = []
        v_loss = []
        for epoch in trange(epochs):
            for step, data in enumerate(self.dataset["train"]):
                input_data = data['KUL'].cuda()
                prediction = self(input_data)
                loss = self.loss_fn(prediction, data['EES'].cuda())
                loss.backward()
                self.optimizer.step()
                self.zero_grad()
                b_loss.append(loss.item())
                #  print(f'Batch loss: {loss.item()}', flush=True)
            if epoch % self.valid_freq and epoch != 0:
                print(f'Validation loss: {self.validation()}', flush=True)
                v_loss.append(self.validation())
        print("Train is complete")

        self.test()
        self.save_weights()
        #  self.save_checkpoint()
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
        average_loss = 0
        step = 0
        for step, data in enumerate(self.dataset['validation']):
            with torch.no_grad():
                input_data = data['KUL'].cuda()
                prediction = self(input_data)
                loss = self.loss_fn(prediction, data['EES'].cuda())
                average_loss += loss.item()
        return average_loss / (step + 1)
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
        print("saveing weights is complete")

    # def save_checkpoint(self):
    #     checkpoint = {'model': self.model,
    #      'state_dict': self.model.state_dict(),
    #      'optimizer' : self.optimizer.state_dict()}
    #    torch.save(checkpoint, 'checkpoint.pth')
    #     print("saveing checkpoint is complete")

    def load_weights(self):
        self.optimizer = torch.load('model.weights')
        print("loading is complete")

    # def load_checkpoint(self, filepath):
    #    checkpoint = torch.load(self, filepath)
    #     self.model = checkpoint['model']
    #     self.model.load_state_dict(checkpoint['state_dict'])
    #     for parameter in self.model.parameters():
    #         parameter.requires_grad = False
    #     self.model.eval()
    #     return self.model

if __name__ == "__main__":
    learner = CarBadnessGuesser()
    torch.backends.cudnn.enabled = False
    learner.train()
#    learner.model = learner.load_checkpoint('checkpoint.pth')
#    learner.load_weights()
#    learner.forward()

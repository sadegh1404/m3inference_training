from m3inference.utils import logging
import argparse
from m3inference.dataset import M3InferenceDataset
from torch.utils.data import DataLoader
from m3inference.full_model import M3InferenceModel 
import json 
import torch 

class M3Trainer:

    def __init__(self,train_data_path,batch_size=16,label='gender',train_split=0.8,validation=False,image_dir='profile_image'):

        self.train_data_path = train_data_path
        self.data = self.read_json_file(train_data_path)
        self.batch_size = batch_size
        self.label_level = label
        self.train_data , self.test_data = self.train_val_test_split(self.data,train_split)

        self.train_data_loader = DataLoader(M3InferenceDataset(self.train_data,label_level = label,image_dir=image_dir),batch_size=self.batch_size)
        
        self.test_data_loader = DataLoader(M3InferenceDataset(self.test_data,label_level = label,image_dir=image_dir),batch_size=self.batch_size)

        self.m3Model = M3InferenceModel()

        self.loss_function = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.m3Model.parameters())

        self.logger = logging.getLogger(__name__)

    def read_json_file(self,json_path):
        with open(json_path,'r') as f:
            file_content = json.loads(f.read())
        return file_content


    def train_val_test_split(self,data,train_split,validation=False):
        len_data = len(data)
        idx_train = int(len_data * train_split)
        len_val_test = (len_data - idx_train) / 2
        if validation:
            return data[:idx_train] , data[idx_train:idx_trian+len_val_test] , data[idx_train+len_val_test:]
        else:
            return data[:idx_train] , data[idx_train:]
        
    
    def train(self,epochs=10):
        for epoch in range(epochs):

            running_loss = 0.0

            for i,batch in enumerate(self.train_data_loader,0):

                x,y = batch 
                y = y.double()
                x,y = x.cuda(),y.cuda()

                self.optimizer.zero_grad()

                outputs = self.m3Model(x,label=self.label_level)

                loss = self.loss_function(outputs,y)
                loss.backward()
                self.optimizer.spet()
                if i%100 == 0: 
                    print("Batch {} Epoch {} Loss {}".format(i,epoch+1,running_loss/i))
                running_loss += loss.item()
                     
            train_log = 'Epoch: {} loss: {}'.format(epoch+1, running_loss/i )
            print(train_log)
            self.logger.info(train_log)
    
    def save_model(self,save_path):
        torch.save(self.m3Model.state_dict(),save_path)
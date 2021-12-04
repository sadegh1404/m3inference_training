from tensorflow import keras
import unicodedata

from PIL import Image

import numpy as np 
from .utils import *
from .consts import *
import os



class DataGenerator(keras.utils.Sequence):

    def __init__(self, data: list, use_img=True,label_level = None,image_dir='.',batch_size=4):

        
        self.use_img = use_img
        self.data = []
        self.label_level = label_level
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.shuffle = True
        for entry in data:
            entry = DotDict(entry)
            if use_img:
                self.data.append([entry.id, entry.lang, normalize_space(str(entry.name)),
                                  normalize_space(str(entry.screen_name)),
                                  normalize_url(normalize_space(str(entry.description))), entry.img_path,entry.gender,entry.age])
            else:
                self.data.append([entry.id, entry.lang, normalize_space(str(entry.name)),
                                  normalize_space(str(entry.screen_name)),
                                  normalize_url(normalize_space(str(entry.description))),entry.gender,entry.age])
        self.on_epoch_end()
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.data[k] for k in indexes]
        lang_input = []
        username_input = []
        screen_name_input = [] 
        des_input = [] 
        image_input = [] 
        y_batch = []
        for dt in list_IDs_temp:
            x,y = self._preprocess_data(dt)
            lang_input.append(x[0])
            username_input.append(x[1])
            screen_name_input.append(x[2])
            des_input.append(x[3])
            image_input.append(x[4])
            y_batch.append(y)
        return np.array(lang_input),np.array(username_input),np.array(screen_name_input),np.array(des_input),np.array(image_input),np.array(y_batch)

    def _preprocess_data(self, data):


        if self.use_img:
            _id, lang, username, screenname, des, img_path , gender,age = data
            # image
            fig = self._image_loader(img_path)
        else:
            _id, lang, username, screenname, des = data

        # text
        lang_tensor = LANGS[lang]

        username_tensor = [0] * USERNAME_LEN
        if username.strip(" ") == "":
            username_tensor[0] = EMB["<empty>"]
            username_len = 1
        else:
            if len(username) > USERNAME_LEN:
                username = username[:USERNAME_LEN]
            username_len = len(username)
            username_tensor[:username_len] = [EMB.get(i, len(EMB) + 1) for i in username]

        screenname_tensor = [0] * SCREENNAME_LEN
        if screenname.strip(" ") == "":
            screenname_tensor[0] = 32
            screenname_len = 1
        else:
            if len(screenname) > SCREENNAME_LEN:
                screenname = screenname[:SCREENNAME_LEN]
            screenname_len = len(screenname)
            screenname_tensor[:screenname_len] = [EMB.get(i, len(EMB) + 1) for i in screenname]

        des_tensor = [0] * DES_LEN
        if des.strip(" ") == "":
            des_tensor[0] = EMB["<empty>"]
            des_len = 1
        else:
            if len(des) > DES_LEN:
                des = des[:DES_LEN]
            des_len = len(des)
            des_tensor[:des_len] = [EMB.get(i, EMB[unicodedata.category(i)]) for i in des]

        if self.label_level == 'gender':
            y = keras.utils.to_categorical(gender_class_mapper[gender],num_classes=2)
        
        elif self.label_level == 'age':
            y = keras.utils.to_categorical(age-1,num_classes=4)
        elif self.label_level =='gender_age':
            y = keras.utils.to_categorical(gender,num_classes=2),keras.utils.to_categorical(age,num_classes=4)
        if self.use_img:
            return [lang_tensor,username_tensor, screenname_tensor,des_tensor,fig],y
        else:
            return [np.array(lang_tensor),np.array(username_tensor), np.array(screenname_tensor),np.array(des_tensor)],y


    def _image_loader(self, image_name):
        try:
            image = Image.open(os.path.join(self.image_dir,image_name))
            image = image.resize((224,224))
            image = image.convert(mode="RGB")
        except:
            image = Image.open(os.path.join(self.image_dir,'default.png'))
            image = image.resize((224,224))
            image = image.convert(mode="RGB")
        
        return image
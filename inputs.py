import tensorflow as tf
import config as cfg
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data

 
class input_pipeline(object):
	def __init__(self):
		self.x_train=None
		self.x_val=None
		self.x_test=None
		self.y_train=None
		self.y_val=None
		self.y_test=None

	def load_label_file(self):		
		(x_train, y_train), (x_test, y_test) = load_data()
	
		return x_train,x_test,y_train,y_test

		
	def split_dataset(self):
		x_train,x_test,y_train,y_test=self.load_label_file()
		x_train, x_val, y_train, y_val=train_test_split(x_train,y_train,stratify=y_train,test_size=0.2, random_state=45)
		return x_train,y_train, x_val,y_val,x_test,y_test
		
		
	def get_dataset(self):
		self.x_train,self.y_train, self.x_val,self.y_val,self.x_test,self.y_test=self.split_dataset()
		return self.x_train,self.y_train,self.x_val,self.y_val,self.x_test,self.y_test
		
	def get(self,batch_size, data, labels,step):
		cursor=batch_size*step
		idx = np.arange(0 , len(data))
		idx = idx[ cursor : cursor+batch_size]
		np.random.shuffle(idx)
		
		data_shuffle=[]
		labels_shuffle=[]
		
		for i in idx:
			image=cv2.resize(data[i],(cfg.image_size, cfg.image_size))
			data_shuffle.append(image/255.0)
			label = np.zeros((cfg.num_class))
			label[labels[i]]=1
			labels_shuffle.append(label)
			
		data_shuffle=np.asarray(data_shuffle)
		labels_shuffle=np.asarray(labels_shuffle)
		return data_shuffle,labels_shuffle 

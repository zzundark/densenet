
import tensorflow as tf
import config as cfg
import inputs as input
import os
import sys
from timer import Timer
import numpy as np

#solver 클래스 : 모델의 학습, 테스트를 관리 
class solver(object):

	def __init__(self,net,input,logs_path,learning_rate=0.0001):
		if not os.path.exists(logs_path):
			os.makedirs(logs_path)
		self.logs_path=logs_path
		self.net = net
		self.input=input
		self.t_vars = tf.trainable_variables()
		self.init_learning_rate=learning_rate
		#cost, train_step, correct_prediction, accuracy, training_flag
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.net.logits, labels=self.net.labels))
		
		"""
		MomentumOptimizer version(original in paper) 
		l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
		optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True)
		self.train_step = optimizer.minimize(cost + l2_loss * weight_decay)
		init_learning_rate = 0.1
		"""		
		self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
		self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
		self.correct_prediction = tf.equal(tf.argmax(self.net.logits, 1), tf.argmax(self.net.labels, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float")) 
		
		#웨이트 저장 
		self.saver = tf.train.Saver(max_to_keep=None,var_list=self.t_vars)
		
		#GPU 설정
		self.config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True))
		self.sess = tf.Session(config=self.config)
	
		#sigle op에 모든 summery merge함 
		self.summary_op = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(logs_path, flush_secs=60)
		self.writer.add_graph(self.sess.graph)	
		
		#초기화 
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
		
		# self.weights_file='./restore/w-8'
		# if self.weights_file is not None:
			# print('Restoring weights from: ' + self.weights_file)
			# self.saver.restore(self.sess, self.weights_file)		
		
		
	#loss와 auccuracy 연산
	def cal_loss_accuracy(self,epoch,total_size,batch_size,data,label,loss_tag,accuracy_tag):
		cal_loss=0
		cal_acc=0
		
		assert (total_size > batch_size), '배치 사이즈(%d)가 데이터 셋 크기(%d)보다 작습니다'%(batch_size,total_size)
		
		total_iteration = int(total_size/batch_size)
		for step in range(total_iteration):
			images,labels = self.input.get(cfg.batch_size,data,label,step)
			feed_dicts = {self.net.images: images,
							self.net.labels: labels,
							self.net.training: False,
							self.net.keep_prob: 1.0
							}
			loss, acc = self.sess.run(
				[self.cost,self.accuracy],
				feed_dict=feed_dicts)
			cal_loss += loss  
			cal_acc += acc
			
		cal_loss/=float(total_iteration)
		cal_acc/=float(total_iteration)
		
		_summary = tf.Summary(value=[tf.Summary.Value(tag=loss_tag, simple_value=cal_loss),
								  tf.Summary.Value(tag=accuracy_tag, simple_value=cal_acc)])

		self.writer.add_summary(_summary,epoch)
		return cal_loss,cal_acc
		
	def train_and_test(self):
		
		train_timer = Timer()
		load_timer = Timer()
		f=open(os.path.join(self.logs_path, 'log.txt'), 'w')
		x_train,y_train, x_val,y_val,x_test,y_test=self.input.get_dataset()
		display_step=1
		max_accuracy=0
		max_index=0
		total_iteraton=len(x_train)
		
		
		epoch_learning_rate = self.init_learning_rate
		for epoch in range(cfg.max_epoch+1):
		
			if epoch == (cfg.max_epoch * 0.5) or epoch == (cfg.max_epoch * 0.75):
				epoch_learning_rate = epoch_learning_rate / 10
			
			total_batch=int(total_iteraton/cfg.batch_size)
			for step in range(total_batch):
				if(epoch==0):
					break
				load_timer.tic()
				images,labels = self.input.get(cfg.batch_size,x_train,y_train,step)
				load_timer.toc()
		
				train_timer.tic()
				feed_dicts = {self.net.images: images,
								self.net.labels: labels,
								self.net.training: True,
								self.net.keep_prob: 0.2,
								self.learning_rate: epoch_learning_rate
								}
				_ = self.sess.run([self.train_step],
								feed_dict=feed_dicts)
				train_timer.toc()	

			# Display logs per epoch step
			if epoch % display_step == 0:	
				train_loss,train_acc=self.cal_loss_accuracy(epoch,
															len(x_train),
															cfg.batch_size,
															x_train,
															y_train,
															"training_loss",
															"training_accuracy")	
				valid_loss,valid_acc=self.cal_loss_accuracy(epoch,
															len(x_val),
															cfg.batch_size,
															x_val,
															y_val,
															"validation_loss",
															"validation_accuracy")
															
				line = ("epoch: %d/%d, train_loss: %.9f, train_acc: %.9f valid_loss: %.9f, valid_acc: %.9f Speed: %.9f s/step  Load: %.9f s/step, Remain: %s\n" 
				%(epoch, cfg.max_epoch, train_loss, train_acc,valid_loss,valid_acc,train_timer.average_time,load_timer.average_time,train_timer.remain(epoch, cfg.max_epoch)))
				print(line)
				f.write(line)
				if valid_acc>=max_accuracy:
					max_accuracy=valid_acc
					max_index=epoch				
				self.saver.save(self.sess,cfg.weight_dir+"/w",epoch)	

		self.saver.restore(self.sess,cfg.weight_dir+"/w-%d"%(max_index))
		test_loss1,test_acc1=self.cal_loss_accuracy(1,len(x_test),cfg.batch_size,x_test,y_test,"test_loss","test_accuracy")
		line = "test_loss: %.9f, test_acc:%.9f,max_index:%d\n"%(test_loss1,test_acc1,max_index)
		print(line)
		f.write(line)
		f.close()
		
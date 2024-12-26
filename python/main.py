import math
import random
import time
import os
import sys
from datetime import datetime

import collections
from collections import namedtuple
from collections import deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
import pymss
from Model import *
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# Episode = namedtuple('Episode',('s','a','r', 'value', 'logprob'))  
# exo and human 
Episode = namedtuple('Episode',('s_e', 'a_e', 'r_e', 'value_e', 'logprob_e', 's_h', 'a_h', 'r_h', 'value_h', 'logprob_h')) 
class EpisodeBuffer(object):
	def __init__(self):
		self.data = []

	def Push(self, *args):
		self.data.append(Episode(*args))
	def Pop(self):
		self.data.pop()
	def GetData(self):
		return self.data
MuscleTransition = namedtuple('MuscleTransition',('JtA','tau_des','L','b'))
class MuscleBuffer(object):
	def __init__(self, buff_size = 10000):
		super(MuscleBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def Push(self,*args):
		self.buffer.append(MuscleTransition(*args))

	def Clear(self):
		self.buffer.clear()
Transition = namedtuple('Transition',('s','a', 'logprob', 'TD', 'GAE'))
class ReplayBuffer(object):
	def __init__(self, buff_size = 10000):
		super(ReplayBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def Push(self,*args):
		self.buffer.append(Transition(*args))

	def Clear(self):
		self.buffer.clear()
class PPO(object):
	def __init__(self,meta_file):
		np.random.seed(seed = int(time.time()))
		self.num_slaves = 16
		self.env = pymss.pymss(meta_file,self.num_slaves)  
  
		self.only_human = 1    # or 0 with human and exo network 

		self.use_muscle = self.env.UseMuscle()    
		self.num_muscles = self.env.GetNumMuscles()    

  		# human training details 
		self.num_human_state = self.env.GetNumState()  
		self.num_human_action = self.env.GetNumAction()  
  
		# exo training details 
		self.num_exo_state = self.num_human_state 
		self.num_exo_action = 2       
		
		self.num_epochs = 10   
		self.num_epochs_muscle = 3  
		self.num_evaluation = 0
		self.num_tuple_so_far = 0
		self.num_episode = 0
		self.num_tuple = 0
		self.num_simulation_Hz = self.env.GetSimulationHz()
		self.num_control_Hz = self.env.GetControlHz()
		self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

		self.gamma = 0.99   
		self.lb = 0.99   

		self.buffer_size = 2048  
		self.batch_size = 128
		self.muscle_batch_size = 128

		self.human_replay_buffer = ReplayBuffer(30000)    
		self.exo_replay_buffer = ReplayBuffer(30000)    
		self.replay_buffer = ReplayBuffer(30000)
		self.muscle_buffer = {}

		# create models  
		self.exo_model = SimulationExoNN(self.num_exo_state,self.num_exo_action)    
		self.human_model = SimulationHumanNN(self.num_human_state,self.num_human_action)  
		self.muscle_model = MuscleNN(self.env.GetNumTotalMuscleRelatedDofs(),self.num_human_action,self.num_muscles)
		if use_cuda:
			self.human_model.cuda()   
			self.muscle_model.cuda()   

		self.default_learning_rate = 1E-4
		self.default_clip_ratio = 0.2
		self.learning_rate = self.default_learning_rate
		self.clip_ratio = self.default_clip_ratio
  
		# optimizer 
		self.optimizer_exo = optim.Adam(self.exo_model.parameters(),lr=self.learning_rate)
		self.optimizer_human = optim.Adam(self.human_model.parameters(),lr=self.learning_rate)
		self.optimizer_muscle = optim.Adam(self.muscle_model.parameters(),lr=self.learning_rate)
		self.max_iteration = 50000  

		self.w_entropy = -0.001

		self.loss_actor = 0.0
		self.loss_critic = 0.0
		self.loss_muscle = 0.0
		self.rewards = []  
		self.rewards_exo = []   
		self.rewards_human = []    
		self.sum_return = 0.0
		self.max_return = -1.0  
		self.max_return_epoch = 1  
		self.tic = time.time()

		self.episodes = [None]*self.num_slaves
		for j in range(self.num_slaves):
			self.episodes[j] = EpisodeBuffer()   
   
		self._action_filter_exo = [self._BuildExoActionFilter() for _i in range(self.num_slaves)]  
		self._action_filter_human = [self._BuildHumanActionFilter() for _i in range(self.num_slaves)]   
  
		self.env.Resets(True)  
	
	def _BuildExoActionFilter(self):
		sampling_rate = self.num_control_Hz #1 / (self.time_step * self._action_repeat)
		num_joints = self.num_exo_action
		a_filter = action_filter.ActionFilterButter(
			sampling_rate=sampling_rate, num_joints=num_joints, filter_low_cut = 0, filter_high_cut = 8)
		return a_filter

	def _BuildHumanActionFilter(self):
		sampling_rate = self.num_control_Hz #1 / (self.time_step * self._action_repeat)
		num_joints = self.num_human_action
		a_filter = action_filter.ActionFilterButter(
			sampling_rate=sampling_rate, num_joints=num_joints, filter_low_cut = 0, filter_high_cut = 5)
		return a_filter

	def _ResetExoActionFilter(self):
		for filter in self._action_filter_exo:  
			filter.reset()
		return  

	def _ResetHumanActionFilter(self):
		for filter in self._action_filter_human:
			filter.reset()  
		return  

	def _FilterExoAction(self, action):   
		for i in range(action.shape[0]):  
			if sum(self._action_filter_exo[i].xhist[0])[0] == 0:
				self._action_filter_exo[i].init_history(action[i])  

		filtered_action_exo = []
		for i in range(action.shape[0]):
			filtered_action_exo.append(self._action_filter_exo[i].filter(action[i]))
		return np.vstack(filtered_action_exo)  

	def _FilterHumanAction(self, action):
		for i in range(action.shape[0]):
			if sum(self._action_filter_human[i].xhist[0])[0] == 0:
				self._action_filter_human[i].init_history(action[i])

		filtered_action_human = []
		for i in range(action.shape[0]):
			filtered_action_human.append(self._action_filter_human[i].filter(action[i]))
		return np.vstack(filtered_action_human)  

	def SaveModel(self):  
		self.exo_model.save('../nn/current_exo.pt')  
		self.human_model.save('../nn/current_human.pt')   
		self.muscle_model.save('../nn/current_muscle.pt')    

		if self.max_return_epoch == self.num_evaluation:
			self.exo_model.save('../nn/max_exo.pt')
			self.human_model.save('../nn/max_human.pt')  
			self.muscle_model.save('../nn/max_muscle.pt')  
		if self.num_evaluation%100 == 0:
			self.exo_model.save('../nn/'+str(self.num_evaluation//100)+'_exo.pt')  
			self.human_model.save('../nn/'+str(self.num_evaluation//100)+'_human.pt')  
			self.muscle_model.save('../nn/'+str(self.num_evaluation//100)+'_muscle.pt')  

	def LoadModel(self,path):  
		self.exo_model.load('../nn/'+path+'_exo.pt')  
		self.human_model.load('../nn/'+path+'_human.pt')  
		self.muscle_model.load('../nn/'+path+'_muscle.pt')   
  
	def LoadExoModel(self,path):  
		self.exo_model.load('../nn/'+path+'_exo.pt')    
  
	def LoadHumanModel(self,path):   
		self.human_model.load('../nn/'+path+'_human.pt')    
  
	def LoadMuscleModel(self,path):   
		self.muscle_model.load('../nn/'+path+'_muscle.pt')    

	def ComputeTDandGAE(self):
		self.exo_replay_buffer.Clear()   
		self.human_replay_buffer.Clear()   
		self.muscle_buffer = {}    

		self.sum_return_exo = 0.0
		self.sum_return_human = 0.0   
		for epi in self.total_episodes:  
			data = epi.GetData()
			size = len(data)
			if size == 0:
				continue
			
			states_exo, actions_exo, rewards_exo, values_exo, logprobs_exo, \
			states_human, actions_human, rewards_human, values_human, logprobs_human = zip(*data)

			values_exo = np.concatenate((values_exo, np.zeros(1)), axis=0)
			advantages_exo = np.zeros(size)  
			ad_exo = 0   
   
			values_human = np.concatenate((values_human, np.zeros(1)), axis=0)
			advantages_human = np.zeros(size)  
			ad_human = 0   

			epi_return_exo = 0.0  
			epi_return_human = 0.0    
			for i in reversed(range(len(data))):
				# for exo  
				epi_return_exo += rewards_exo[i]
				delta_exo = rewards_exo[i] + values_exo[i+1] * self.gamma - values_exo[i]
				ad_exo = delta_exo + self.gamma * self.lb * ad_exo
				advantages_exo[i] = ad_exo
    
				# for human 
				epi_return_human += rewards_human[i]   
				delta_human = rewards_human[i] + values_human[i+1] * self.gamma - values_human[i]
				ad_human = delta_human + self.gamma * self.lb * ad_human
				advantages_human[i] = ad_human
    
			self.sum_return_exo += epi_return_exo
			self.sum_return_human += epi_return_human  

			TD_exo = values_exo[:size] + advantages_exo  
			TD_human = values_human[:size] + advantages_human  
			
			for i in range(size):
				self.exo_replay_buffer.Push(states_exo[i], actions_exo[i], logprobs_exo[i], TD_exo[i], advantages_exo[i]) 
				self.human_replay_buffer.Push(states_human[i], actions_human[i], logprobs_human[i], TD_human[i], advantages_human[i])   
	
		self.num_episode = len(self.total_episodes)
		self.num_tuple = len(self.replay_buffer.buffer)
		print('SIM : {}'.format(self.num_tuple))
		self.num_tuple_so_far += self.num_tuple

		# muscle calculation 
		self.env.ComputeMuscleTuples()  

		self.muscle_buffer['JtA'] = self.env.GetMuscleTuplesJtA()
		self.muscle_buffer['TauDes'] = self.env.GetMuscleTuplesTauDes()
		self.muscle_buffer['L'] = self.env.GetMuscleTuplesL()
		self.muscle_buffer['b'] = self.env.GetMuscleTuplesb()
		print(self.muscle_buffer['JtA'].shape)
  
	def GenerateTransitions(self):   
		self.total_episodes = []  
  
		# total model  
		states = [None]*self.num_slaves
		actions = [None]*self.num_slaves 
  
		rewards = [None]*self.num_slaves
		states_next = [None]*self.num_slaves    
  
		# exo  
		rewards_exo = [None]*self.num_slaves  
		rewards_human = [None]*self.num_slaves   
  
		if self.only_human: 
			# only human model 
			states = self.env.GetStates()   
			
			states_exo = states  
			states_human = states  
		else:   
			# get seperate states  
			# states = self.env.GetFullObservations()    
	
			# states_exo = states[:, :-self.num_human_state]     
			# states_human = states[:, -self.num_human_state:]    
			states_exo = self.env.GetExoStates()   
			states_human = self.env.GetHumanStates()     
  
		local_step = 0  
		terminated = [False]*self.num_slaves
		counter = 0
		while True:
			counter += 1
			if counter%10 == 0:
				print('SIM : {}'.format(local_step),end='\r') 
    
			# human action 
			a_dist_human,v_human = self.human_model(Tensor(states_human))
			actions_human = a_dist_human.sample().cpu().detach().numpy()    
			logprobs_human = a_dist_human.log_prob(Tensor(actions_human)).cpu().detach().numpy().reshape(-1)
			values_human = v_human.cpu().detach().numpy().reshape(-1)   
   
   			# exo action 
			a_dist_exo,v_exo = self.exo_model(Tensor(states_exo))      
			actions_exo = a_dist_exo.sample().cpu().detach().numpy()     
			logprobs_exo = a_dist_exo.log_prob(Tensor(actions_exo)).cpu().detach().numpy().reshape(-1)
			values_exo = v_exo.cpu().detach().numpy().reshape(-1)     

			# set actions of exo and human   
			self.env.SetExoHumanActions(actions_exo, actions_human)    
   
			# update action buffer  
			self.env.UpdateExoActionBuffers(actions_exo)    
			self.env.UpdateHumanActionBuffers(actions_human)     

			# use muscle 
			if self.use_muscle:  
				mt = Tensor(self.env.GetMuscleTorques())    
				for i in range(self.num_simulation_per_control//2):   
					dt = Tensor(self.env.GetDesiredTorques())  
					activations = self.muscle_model(mt,dt).cpu().detach().numpy()  
					self.env.SetActivationLevels(activations)  
					self.env.Steps(2)  
     
				self.env.UpdateStateBuffers()    # update state buffer  
			else:
				self.env.StepsAtOnce()         

				# state buffer update  
				self.env.UpdateStateBuffers()    # update state buffer 

			for j in range(self.num_slaves):   
				nan_occur = False    
				terminated_state = True    

				if np.any(np.isnan(states_exo[j])) or np.any(np.isnan(actions_exo[j])) or np.any(np.isnan(values_exo[j])) or np.any(np.isnan(logprobs_exo[j])) or\
					np.any(np.isnan(states_human[j])) or np.any(np.isnan(actions_human[j])) or np.any(np.isnan(values_human[j])) or np.any(np.isnan(logprobs_human[j])):    
					nan_occur = True
				
				elif self.env.IsEndOfEpisode(j) is False:
					terminated_state = False 
					
					rewards_exo[j] = self.env.GetExoReward(j)  
					rewards_human[j] = self.env.GetHumanReward(j)   
					self.episodes[j].Push(states_exo[j], actions_exo[j], rewards_exo[j], values_exo[j], logprobs_exo[j], \
																states_human[j], actions_human[j], rewards_human[j], values_human[j], logprobs_human[j])  
					local_step += 1   

				if terminated_state or (nan_occur is True):
					if (nan_occur is True):
						self.episodes[j].Pop()  
					self.total_episodes.append(self.episodes[j])
					self.episodes[j] = EpisodeBuffer()   

					self.env.Reset(True,j)  
					
					# print("exo shape :", self.env.GetExoActions(j))
					# self._action_filter_exo[j].init_history(self.env.GetExoActions(j)[:self.num_exo_action])   
					# self._action_filter_human[j].init_history(self.env.GetHumanActions(j)[:self.num_human_action])       

			if local_step >= self.buffer_size: 
				break
			
			if self.only_human:  
				states = self.env.GetStates()   
				states_exo = states  
				states_human = states  
			else:  
				# states = self.env.GetFullObservations()    
				# states_exo = states[:,0:-self.num_human_state]     
				# states_human = states[:,-self.num_human_state:]     
				states_exo = self.env.GetExoStates()   
				states_human = self.env.GetHumanStates()    

	def OptimizeSimulationHumanNN(self):  
		all_transitions = np.array(self.human_replay_buffer.buffer, dtype=object)     
		for j in range(self.num_epochs):
			np.random.shuffle(all_transitions)
			for i in range(len(all_transitions)//self.batch_size):
				transitions = all_transitions[i*self.batch_size:(i+1)*self.batch_size]
				batch = Transition(*zip(*transitions))

				stack_s = np.vstack(batch.s).astype(np.float32)
				stack_a = np.vstack(batch.a).astype(np.float32)
				stack_lp = np.vstack(batch.logprob).astype(np.float32)
				stack_td = np.vstack(batch.TD).astype(np.float32)
				stack_gae = np.vstack(batch.GAE).astype(np.float32)
				
				a_dist,v = self.human_model(Tensor(stack_s))
				'''Critic Loss'''
				loss_critic = ((v-Tensor(stack_td)).pow(2)).mean()
				
				'''Actor Loss'''
				ratio = torch.exp(a_dist.log_prob(Tensor(stack_a))-Tensor(stack_lp))
				stack_gae = (stack_gae-stack_gae.mean())/(stack_gae.std()+ 1E-5)
				stack_gae = Tensor(stack_gae)
				surrogate1 = ratio * stack_gae
				surrogate2 = torch.clamp(ratio,min =1.0-self.clip_ratio,max=1.0+self.clip_ratio) * stack_gae
				loss_actor = - torch.min(surrogate1,surrogate2).mean()
				'''Entropy Loss'''
				loss_entropy = - self.w_entropy * a_dist.entropy().mean()

				self.loss_actor = loss_actor.cpu().detach().numpy().tolist()
				self.loss_critic = loss_critic.cpu().detach().numpy().tolist()
				
				loss = loss_actor + loss_entropy + loss_critic

				self.optimizer_human.zero_grad()    
				loss.backward(retain_graph=True)   
				for param in self.human_model.parameters():  
					if param.grad is not None:  
						param.grad.data.clamp_(-0.5,0.5)  
				self.optimizer_human.step()   
			print('Optimizing sim nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')
  
	def OptimizeSimulationExoNN(self): 
		all_transitions = np.array(self.exo_replay_buffer.buffer,dtype=object)  
		for j in range(self.num_epochs): 
			np.random.shuffle(all_transitions) 
			for i in range(len(all_transitions)//self.batch_size):
				transitions = all_transitions[i*self.batch_size:(i+1)*self.batch_size]
				batch = Transition(*zip(*transitions))  

				stack_s = np.vstack(batch.s).astype(np.float32)

				stack_a = np.vstack(batch.a).astype(np.float32)
				stack_lp = np.vstack(batch.logprob).astype(np.float32)
				stack_td = np.vstack(batch.TD).astype(np.float32)
				stack_gae = np.vstack(batch.GAE).astype(np.float32)
				
				a_dist,v = self.exo_model(Tensor(stack_s))  
				'''Critic Loss'''
				loss_critic = ((v-Tensor(stack_td)).pow(2)).mean()
				
				'''Actor Loss'''
				ratio = torch.exp(a_dist.log_prob(Tensor(stack_a))-Tensor(stack_lp))
				stack_gae = (stack_gae-stack_gae.mean())/(stack_gae.std()+ 1E-5)
				stack_gae = Tensor(stack_gae)
				surrogate1 = ratio * stack_gae
				surrogate2 = torch.clamp(ratio,min =1.0-self.clip_ratio,max=1.0+self.clip_ratio) * stack_gae
				loss_actor = - torch.min(surrogate1,surrogate2).mean()
				'''Entropy Loss'''
				loss_entropy = - self.w_entropy * a_dist.entropy().mean()

				self.loss_actor = loss_actor.cpu().detach().numpy().tolist()
				self.loss_critic = loss_critic.cpu().detach().numpy().tolist()
				
				loss = loss_actor + loss_entropy + loss_critic

				self.optimizer_exo.zero_grad()   
				loss.backward(retain_graph=True)   
				for param in self.exo_model.parameters():   
					if param.grad is not None:  
						param.grad.data.clamp_(-0.5,0.5)  
				self.optimizer_exo.step()   
			print('Optimizing sim nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')  

	def generate_shuffle_indices(self, batch_size, minibatch_size):
		n = batch_size
		m = minibatch_size
		p = np.random.permutation(n)

		r = m - n%m  
		if r>0:
			p = np.hstack([p,np.random.randint(0,n,r)])

		p = p.reshape(-1,m)   
		return p

	def OptimizeMuscleNN(self):  
		for j in range(self.num_epochs_muscle):  
			minibatches = self.generate_shuffle_indices(self.muscle_buffer['JtA'].shape[0],self.muscle_batch_size)

			for minibatch in minibatches:
				stack_JtA = self.muscle_buffer['JtA'][minibatch].astype(np.float32)
				stack_tau_des =  self.muscle_buffer['TauDes'][minibatch].astype(np.float32)
				stack_L = self.muscle_buffer['L'][minibatch].astype(np.float32)
				stack_L = stack_L.reshape(self.muscle_batch_size,self.num_human_action,self.num_muscles)
				stack_b = self.muscle_buffer['b'][minibatch].astype(np.float32)

				stack_JtA = Tensor(stack_JtA)
				stack_tau_des = Tensor(stack_tau_des)
				stack_L = Tensor(stack_L)
				stack_b = Tensor(stack_b)

				activation = self.muscle_model(stack_JtA,stack_tau_des)
				tau = torch.einsum('ijk,ik->ij',(stack_L,activation)) + stack_b

				loss_reg = (activation).pow(2).mean()
				loss_target = (((tau-stack_tau_des)/100.0).pow(2)).mean()

				loss = 0.01*loss_reg + loss_target
				# loss = loss_target

				self.optimizer_muscle.zero_grad()
				loss.backward(retain_graph=True)
				for param in self.muscle_model.parameters():
					if param.grad is not None:
						param.grad.data.clamp_(-0.5,0.5)  
				self.optimizer_muscle.step()   

			print('Optimizing muscle nn : {}/{}'.format(j+1,self.num_epochs_muscle),end='\r')
		self.loss_muscle = loss.cpu().detach().numpy().tolist()
		print('')  
  
	def OptimizeModel(self):
		self.ComputeTDandGAE() 
		self.OptimizeSimulationHumanNN()  
		self.OptimizeSimulationExoNN()  
		if self.use_muscle:  
			self.OptimizeMuscleNN()  
		
	def Train(self):   
		self.GenerateTransitions()   
		self.OptimizeModel()  

	def Evaluate(self):
		self.num_evaluation = self.num_evaluation + 1
		h = int((time.time() - self.tic)//3600.0)
		m = int((time.time() - self.tic)//60.0)
		s = int((time.time() - self.tic))
		m = m - h*60
		s = int((time.time() - self.tic))
		s = s - h*3600 - m*60
		if self.num_episode == 0:
			self.num_episode = 1
		if self.num_tuple == 0:
			self.num_tuple = 1
		if self.max_return < self.sum_return/self.num_episode:
			self.max_return = self.sum_return/self.num_episode
			self.max_return_epoch = self.num_evaluation
		print('# {} === {}h:{}m:{}s ==='.format(self.num_evaluation,h,m,s))
		print('||Loss Actor               : {:.4f}'.format(self.loss_actor))
		print('||Loss Critic              : {:.4f}'.format(self.loss_critic))
		print('||Loss Muscle              : {:.4f}'.format(self.loss_muscle))
		print('||Noise                    : {:.3f}'.format(self.human_model.log_std.exp().mean()))		
		print('||Num Transition So far    : {}'.format(self.num_tuple_so_far))
		print('||Num Transition           : {}'.format(self.num_tuple))
		print('||Num Episode              : {}'.format(self.num_episode))
		print('||Avg Return per episode   : {:.3f}'.format(self.sum_return/self.num_episode))
		print('||Avg Reward per transition: {:.3f}'.format(self.sum_return/self.num_tuple))
		print('||Avg Step per episode     : {:.1f}'.format(self.num_tuple/self.num_episode))
		print('||Max Avg Retun So far     : {:.3f} at #{}'.format(self.max_return,self.max_return_epoch))
		self.rewards.append(self.sum_return/self.num_episode)
		
		self.SaveModel()
		
		print('=============================================')
		return np.array(self.rewards)

import matplotlib
import matplotlib.pyplot as plt

plt.ion()

def Plot(y,title,num_fig=1,ylim=True):
	temp_y = np.zeros(y.shape)
	if y.shape[0]>5:
		temp_y[0] = y[0]
		temp_y[1] = 0.5*(y[0] + y[1])
		temp_y[2] = 0.3333*(y[0] + y[1] + y[2])
		temp_y[3] = 0.25*(y[0] + y[1] + y[2] + y[3])
		for i in range(4,y.shape[0]):
			temp_y[i] = np.sum(y[i-4:i+1])*0.2

	plt.figure(num_fig)
	plt.clf()
	plt.title(title)
	plt.plot(y,'b')
	
	plt.plot(temp_y,'r')

	plt.show()
	if ylim:
		plt.ylim([0,1])
	plt.pause(0.001)

import argparse
import os
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--model',help='model path')
	parser.add_argument('-d','--meta',help='meta file')

	args =parser.parse_args()
	if args.meta is None:
		print('Provide meta file')
		exit()

	ppo = PPO(args.meta)
	nn_dir = '../nn'
	if not os.path.exists(nn_dir):  
		os.makedirs(nn_dir)    
		
	if args.model is not None: 
		ppo.LoadModel(args.model)  
	else:
		ppo.SaveModel()   

	print('num states: {}, num actions: {}'.format(ppo.env.GetNumState(),ppo.env.GetNumAction()))
	for i in range(ppo.max_iteration-5):
		ppo.Train()  
		rewards = ppo.Evaluate()   
		Plot(rewards,'reward',0,False)    
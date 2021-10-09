import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class EOI_Net(nn.Module):
	def __init__(self, obs_len, n_agent):
		super(EOI_Net, self).__init__()
		self.fc1 = nn.Linear(obs_len, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, n_agent)

	def forward(self, x):
		y = F.relu(self.fc1(x))
		y = F.relu(self.fc2(y))
		y = F.softmax(self.fc3(y),dim=1)
		return y

class IVF(nn.Module):
	def __init__(self, obs_len, n_action):
		super(IVF, self).__init__()
		self.fc1 = nn.Linear(obs_len, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, n_action)

	def forward(self, x):
		y = F.relu(self.fc1(x))
		y = F.relu(self.fc2(y))
		y = self.fc3(y)
		return y

class EOI_Trainer(object):
	def __init__(self, eoi_net, ivf, ivf_tar, n_agent, n_feature):
		super(EOI_Trainer, self).__init__()
		self.gamma = 0.92
		self.tau = 0.995
		self.n_agent = n_agent
		self.n_feature = n_feature
		self.eoi_net = eoi_net
		self.ivf = ivf
		self.ivf_tar = ivf_tar
		self.optimizer_eoi = optim.Adam(self.eoi_net.parameters(), lr = 0.0001)
		self.optimizer_ivf = optim.Adam(self.ivf.parameters(), lr = 0.0001)

	def train(self,O,O_Next,A,D):

		O = torch.Tensor(O).cuda()
		O_Next = torch.Tensor(O_Next).cuda()
		A = torch.Tensor(A).cuda().long()
		D = torch.Tensor(D).cuda()

		X = O_Next[:,0:self.n_feature]
		Y = O_Next[:,self.n_feature:self.n_feature+self.n_agent]
		p = self.eoi_net(X)
		loss_1 = -(Y*(torch.log(p+1e-8))).mean() - 0.1*(p*(torch.log(p+1e-8))).mean()
		self.optimizer_eoi.zero_grad()
		loss_1.backward()
		self.optimizer_eoi.step()

		I = O[:,self.n_feature:self.n_feature+self.n_agent].argmax(axis = 1,keepdim=True).long()
		r = self.eoi_net(O[:,0:self.n_feature]).gather(dim=-1,index=I)

		q_intrinsic = self.ivf(O)
		tar_q_intrinsic = q_intrinsic.clone().detach()
		next_q_intrinsic = self.ivf_tar(O_Next).max(axis = 1,keepdim=True)[0]
		next_q_intrinsic = r*10 + self.gamma*(1-D)*next_q_intrinsic
		tar_q_intrinsic.scatter_(dim=-1,index=A,src=next_q_intrinsic)
		loss_2 = (q_intrinsic - tar_q_intrinsic).pow(2).mean()
		self.optimizer_ivf.zero_grad()
		loss_2.backward()
		self.optimizer_ivf.step()

		with torch.no_grad():
			for p, p_targ in zip(self.ivf.parameters(), self.ivf_tar.parameters()):
				p_targ.data.mul_(self.tau)
				p_targ.data.add_((self.tau) * p.data)

class EOI_Trainer_Wrapper(object):
	def __init__(self, eoi_trainer, n_agent, n_feature, max_step, batch_size):
		super(EOI_Trainer_Wrapper, self).__init__()
		self.batch_size = batch_size
		self.n_agent = n_agent
		self.o_t = np.zeros((batch_size*n_agent*(max_step + 1),n_feature+n_agent))
		self.next_o_t = np.zeros((batch_size*n_agent*(max_step + 1),n_feature+n_agent))
		self.a_t = np.zeros((batch_size*n_agent*(max_step + 1),1),dtype=np.int32)
		self.d_t = np.zeros((batch_size*n_agent*(max_step + 1),1))
		self.eoi_trainer = eoi_trainer
	
	def train_batch(self,episode_sample):
		episode_obs = np.array(episode_sample["obs"])
		episode_actions = np.array(episode_sample["actions"])
		episode_terminated = np.array(episode_sample["terminated"])
		ind = 0
		for k in range(self.batch_size):
			for j in range(episode_obs.shape[1]-2):
				for i in range(self.n_agent):
					agent_id = np.zeros(self.n_agent)
					agent_id[i] = 1
					self.o_t[ind] = np.hstack((episode_obs[k][j][i],agent_id))
					self.next_o_t[ind] = np.hstack((episode_obs[k][j+1][i],agent_id))
					self.a_t[ind] = episode_actions[k][j][i]
					self.d_t[ind] = episode_terminated[k][j]
					ind += 1
				if self.d_t[ind-1] == 1:
					break
		for k in range(int((ind-1)/256)):
			self.eoi_trainer.train(self.o_t[k*256:(k+1)*256],self.next_o_t[k*256:(k+1)*256],self.a_t[k*256:(k+1)*256],self.d_t[k*256:(k+1)*256])


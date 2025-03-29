import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader, TensorDataset

def load_data(filname='data.npz'):
	data=np.load(filname, allow_pickle=True)
	train_data = data['train']
	submit_data = data['submit']
	return train_data, submit_data


def comp_disconected_returns(rewards, gamma=0.99):
	T=len(rewards)
	G=np.zeros(T)
	G[-1]=rewards[-1]
	for t in range(T-2,-1,-1):
		G[t]=rewards[t]+gamma*G[t+1]
	return G

def pepare_traning_data(train_episodes,gamma=0.99):
	X_list=[]
	Y_list=[]
	for eposode in train_episodes:
		rewards = eposode['rewards']
		obs = eposode['observations']
		if not isinstance(obs,np.ndarray):
			obs=np.array(obs)
		if not isinstance(rewards, np.ndarray):
			rewards = np.array(rewards)

		T=rewards.shape[0]
		G=comp_disconected_returns(rewards,gamma)

		for t in range(T):
			state = obs[t]
			flat_state = np.array(state).flatten()
			X_list.append(flat_state)
			Y_list.append(G[t])
	X=np.array(X_list)
	y=np.array(Y_list)
	return X,y


class ValueNetwork(nn.Module):
	def __init__(self, input_dim):
		super(ValueNetwork, self).__init__()
		self.net=nn.Sequential(
			nn.Linear(input_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 1)
		)
	def forward(self,x):
		return self.net(x)


def tran_maodel(model,X,y, num_eph=20,batch_size=256,lr=0.001):
	device=torch.device("cpu")
	model.to(device)
	criterion=nn.MSELoss()
	optimizer=optim.Adam(model.parameters(), lr=lr)

	x_tensor = torch.tensor(X, dtype=torch.float32)
	y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

	dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
	dataloader =DataLoader(dataset, batch_size=batch_size, shuffle=True)

	for eph in range(num_eph):
		eph_loss=0.0
		for batch_x, batch_y in dataloader:
			batch_x=batch_x.to(device)
			batch_y = batch_y.to(device)


			optimizer.zero_grad()
			outputs=model(batch_x)
			loss = criterion(outputs,batch_y)
			loss.backward()
			optimizer.step()
			eph_loss+=loss.item()*batch_x.size(0)

		print(f"Epoch {eph + 1}/{num_eph}, Loss = {eph_loss:.4f}")


def predict_submit(model,submit_data):
	model.eval()
	predictions=[]
	device=torch.device("cpu")
	with torch.no_grad():
		for item in submit_data:
			idx=item['idx']
			state=item['observations']
			state=np.array(state)
			flat_state=state.flatten()
			x_tensor=torch.tensor(flat_state,dtype=torch.float32)
			v_pred = model(x_tensor).item()
			predictions.append({"idx": idx, "value": v_pred})
	return predictions


def save_pred(predictions,filrname="answ/submit.json"):
	with open(filrname,"w",encoding="utf-8") as f:
		json.dump(predictions,f,indent=2,ensure_ascii=False)

train_episodes,submit_data=load_data("data.npz")

X,y=pepare_traning_data(train_episodes,gamma=0.99)
input_dim=X.shape[1]
model=ValueNetwork(input_dim)
tran_maodel(model,X,y, num_eph=40,batch_size=256,lr=0.001)

save_pred(predict_submit(model,submit_data))
print("uraaa")








#
#
# import numpy as np
#
# data=np.load('data.npz',allow_pickle=True)
# episodes=data['train']
# test_episodes=data['submit']
#
# gamma=0.99
#
# train_obs=[]
# train_values=[]
# for eposode in episodes:
# 	rewards=np.array(eposode['rewards'])
# 	observation=np.array(eposode['observations'])
#
# 	r=rewards.sum(axis=1)
# 	T=r.shape[0]
# 	G=np.zeros(T)
# 	for t in reversed(range(T-1)):
# 		G[t]=r[t]+gamma*G[t+1]
#
# 	for t in range(T):
# 		state_t=observation[t]
# 		state_t_flat=state_t.flatten()
#
# 		train_obs.append(state_t_flat)
# 		train_values.append(G[t])
#
# train_obs=np.array(train_obs)
# train_values=np.array(train_values)
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# input_dim=train_obs.shape[1]
# hidden_dim=128
# num_eph=10
# batch_aize=256
# lr=1e-3
#
# model=nn.Sequential(
# 	nn.Linear(input_dim,hidden_dim),
# 	nn.ReLU(),
# 	nn.Linear(input_dim,hidden_dim),
# 	nn.ReLU(),
# 	nn.Linear(hidden_dim,1)
# )
#
# optimizer=optim.Adam(model.parameters(),lr=lr)
# loss_fn=nn.MSELoss()
#
# X=torch.tensor(train_obs,dtype=torch.float32)
# Y=torch.tensor(train_values,dtype=torch.float32).view(-1,1)
#
# dataset=torch.utils.data.TensorDataset(X,Y)
# dataloader=torch.utils.data.DataLoader(dataset,batch_aize=batch_aize,shuffle=True)
#
# for eph in range(num_eph):
# 	for batch_x, batch_y in dataloader:
# 		optimizer.zero_grad()
# 		pred=model(batch_x)
# 		loss=loss_fn(pred,batch_y)
# 		loss.backward()
# 		optimizer.step()
#
# 	print(f"Epoch {eph+1}/{num_eph}, Loss = {loss.item():.4f}")
#
# import json
#
# predictions=[]
# for i, test_episode in enumerate(test_episodes):
# 	test_observations=test_episode['observations']
# 	T_test=test_observations.shae[0]-1
#
#
# for i in range(T_test):
# 	state_t=test_observations[t].flatten()
# 	state_t_tensor=torch.tensor(state_t,dtype=torch.float32)
# 	with torch.no_grad():
# 		v_pred=model(state_t_tensor).item()
#
# 	predictions.append({"episode_index": i,"time_step": t, "v_pred": v_pred})
#
# with open("answ/submit.json","w",encoding="utf-8") as f:
# 	json.dump(predictions,f,indent=2,ensure_ascii=False)
#


import argparse
import os
import torch
import torch.optim
from torch.utils.data import DataLoader
import Myloss
import dataloader
import model
from PIL import Image
from MGDB_MDTA_GDFN_CVPR2022 import MGDB
import numpy as np
#writer = SummaryWriter('runs/MGDB-Zero-DCE')


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)





def train(config):

	os.environ['CUDA_VISIBLE_DEVICES']='0'
	#print(config)
	DCE_net = model.EnhanceNet().cuda()
	DCE_net.apply(weights_init)

	if config.load_pretrain == True:
		DCE_net.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)



	L_color = Myloss.L_color()
	L_spa = Myloss.L_spa()
	L_exp = Myloss.L_exp(16,0.6)
	L_TV = Myloss.L_TV()

	# # contrastive loss
	# L_CL = Myloss.ContrastiveLoss()
	# # Contrastive  datasets
	# negative_path = "Zero-DCE_code/data/Contrast/low/" # negative samples
	# #print(negative_path)
	# negative_path_dir = os.listdir(negative_path)
	# positive_path = "Zero-DCE_code/data/Contrast/GT/"
	# #print(positive_path)
	# positive_path_dir = os.listdir(positive_path)

	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

	DCE_net.train()
	#for epoch in tqdm.tqdm(range(config.num_epochs)):
	for epoch in range(config.num_epochs):
		print('epoch:', epoch)
		for iteration, img_lowlight in enumerate(train_loader):

			img_lowlight = img_lowlight.cuda()

			enhanced_image_1,enhanced_image,A  = DCE_net(img_lowlight)

			# contrastive loss

			# sample_n = random.sample(negative_path_dir, 8)
			# images_n = []
			# for path in sample_n:
			# 	img_n = Image.open(negative_path + path)
			# 	img_n = img_n.resize((256, 256), Image.Resampling.LANCZOS)
			# 	#img_n = (np.asarray(img_n)/255.0)
			# 	img_n_tensor = transforms.ToTensor()(img_n)
			# 	img_n_tensor = img_n_tensor.unsqueeze(0)
			# 	images_n.append(img_n_tensor)
			# n = torch.cat(images_n, 0)
			# n = n.cuda()

			# sample_p = random.sample(positive_path_dir, 8)
			# images_p = []
			# for path in sample_p:
			# 	img_p = Image.open(positive_path + path)
			# 	img_p = img_p.resize((256, 256), Image.Resampling.LANCZOS)
			# 	#img_p = (np.asarray(img_p) / 255.0)
			# 	img_p_tensor = transforms.ToTensor()(img_p)
			# 	img_p_tensor = img_p_tensor.unsqueeze(0)
			# 	images_p.append(img_p_tensor)
			# p = torch.cat(images_p, 0)
			# p = p.cuda()
			# loss_CL = L_CL(enhanced_image, p, n)
			Loss_TV = 200*L_TV(A)
			loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
			loss_col = 5*torch.mean(L_color(enhanced_image))
			loss_exp = 10*torch.mean(L_exp(enhanced_image))
			#print(f"features dtype: {enhanced_image.dtype}, positive_samples dtype: {p.dtype}, negative_samples dtype: {n.dtype}")


			loss =  Loss_TV + loss_spa + loss_col + loss_exp
			#writer.add_scalar("loss", loss.item(), iteration + 1)


			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), config.grad_clip_norm)

			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				#print("loss_CL at iteration", iteration+1, ":", loss_CL.item())
				#print("loss1 at iteration", iteration+1, ":", loss1.item())
				print("Loss at iteration", iteration+1, "Total Loss:", loss.item())
				#print("features",enhanced_image.shape)
			if ((iteration+1) % config.snapshot_iter) == 0:
				if not os.path.exists(config.snapshots_folder):
					os.makedirs(config.snapshots_folder)
				torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

#writer.close()



if __name__ == "__main__":

	#torch.cuda.empty_cache()

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="./data/reflectance_train_data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=16)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	# snapshots/Zero-DCE-CL
	parser.add_argument('--snapshots_folder', type=str, default="./snapshots/re_zero-DCE/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "./Zero-DCE_code/snapshots/Epoch99.pth")
	parser.add_argument('--contrastive_images_path', type=str, default="./Zero-DCE_code/data/Contrast/" )
	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)








	

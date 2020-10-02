from invtf.datasets import *
from invtf.models 	import *
import argparse
import os 

"""
	DISCLAIMER: 

		This is just a placeholder file, the current architectures are substantially smaller than that
		described in their respective articles. Furthermore, some functionality has not been implemented
		yet, the results are thus substantially worse than that reported in the original articles. 

"""

if __name__ == "__main__": 

	parser = argparse.ArgumentParser()

	parser.add_argument("--problem", type=str, default='mnist', help="Problem (mnist/cifar10")  # add fmnist, cifar100, celeb, svnh, rooms. 
	parser.add_argument("--model", 	 type=str, default='nice',  help="Model (nice/realnvp/glow/flow++/iresnet")  # add fmnist, cifar100, celeb, svnh, rooms. 
		
	args = parser.parse_args()


	# Load Dataset (refactor into just giving args.problem and we get data/imgshape) 
	if args.problem == "mnist": 
		X = mnist()
		img_shape = X.shape[1:-1]
	if args.problem in ["fmnist", "fashion_mnist","fashionmnist", "fasion-mnist"]: 
		X = fmnist()
		img_shape = X.shape[1:-1]
	if args.problem in ["cifar10", "cifar"]:  
		X = cifar10()
		img_shape = X.shape[1:]
	if args.problem == "cifar100":  
		X = cifar100()
		img_shape = X.shape[1:]

	# TO BE IMPLEMENTED
	if args.problem in ["celeba", "celeb", "faces"]:  
		X = celeba(n=5000, resolution=32)
		print(X.shape)
		img_shape = X.shape[1:]
	if args.problem == "imagenet32":  raise NotImplementedError()

	# Get Model. 
	if args.model == "nice":  	
		X = X.reshape((X.shape[0], -1))
		g = invtf.models.NICE.model(X)
	if args.model == "realnvp": g = invtf.models.RealNVP.model(X)
	if args.model == "glow":	g = invtf.models.Glow.model(X)
	if args.model == "flowpp":  g = invtf.models.FlowPP.model(X)
	if args.model == "conv3d":  g = invtf.models.Conv3D.model(X)
	if args.model == "iresnet": raise NotImplementedError()

	# Print summary of model. 
	g.summary()

	# Initialize plots 
	fig_rec, 	ax_rec 		= plt.subplots(1, 3)
	fig_fakes, 	ax_fakes 	= plt.subplots(5, 5) 
	fig_loss, 	ax_loss 	= plt.subplots()
	# OBS: Show plots on second screen, remove if you only use one screen. 
	fig_rec.	canvas.manager.window.wm_geometry("+2000+0")
	fig_fakes.	canvas.manager.window.wm_geometry("+2600+0")
	fig_loss.	canvas.manager.window.wm_geometry("+3200+0")

	for i in range(3): ax_rec[i].axis("off")
	for k in range(5): 
		for l in range(5):
			ax_fakes[k,l].axis('off')


	# Initialize folder to save training history plots. 
	histories = {}
	folder_path = "reproduce/" + args.problem + "_" + args.model + "/"
	if not os.path.exists(folder_path): os.makedirs(folder_path)


	# Train model for epochs iterations. 
	epochs = 100
	for i in range(epochs): 

		history = g.fit(X, batch_size=128, epochs=1)	

		# Init histories to fit the history object. 
		if histories == {}: 
			for key in history.history.keys(): histories[key] = []
				
		# Update loss plot. 
		ax_loss.cla()
		for j, key in enumerate(history.history.keys()): 
			histories[key] += history.history[key]
			ax_loss.plot(np.arange(len(histories[key])), histories[key], "C%i"%j, label=key)

		ax_loss.legend()
		ax_loss.set_ylabel("NLL")
		ax_loss.set_xlabel("Epochs")
		ax_loss.set_ylim([0, 10])
		ax_loss.set_xlim([0, epochs])
		ax_loss.set_title(args.problem + " " + args.model)
		
		# Plot fake/real/reconstructed image. 
		fake = g.sample(1, fix_latent=True, std=0.7)

		ax_rec[0].imshow(fake.reshape(img_shape)/255)
		ax_rec[0].set_title("Fake")

		ax_rec[1].imshow(X[0].reshape(img_shape)/255)
		ax_rec[1].set_title("Real")

		ax_rec[2].imshow(g.rec(X[:1]).reshape(img_shape)/255)
		ax_rec[2].set_title("Reconstruction")
		if i == 0: 
			fig_rec.tight_layout()
			fig_fakes.tight_layout()


		stds = [1.0, 0.8, 0.6, 0.5, 0.4] 
		for k in range(5): 
			current_std = stds[k]
			fakes 		= g.sample(5, fix_latent=True, std=current_std)
			ax_fakes[0, k].set_title( current_std )
			for l in range(5): 
				ax_fakes[k,l].imshow(fakes[l].reshape(img_shape)/255)

		plt.pause(.1)

		fig_fakes.savefig(folder_path + "img_%i.png"%(i+1))
		fig_loss.savefig(folder_path + "loss.png")
		np.savez(folder_path + "imgs_%i.npz"%(i+1), fake)


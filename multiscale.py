import invtf

# Load data
X = invtf.datasets.cifar10()
input_shape = X.shape[1:]

# Model
g = invtf.Generator()

# Pre-process
g.add(invtf.dequantize.UniformDequantize(input_shape=input_shape)) 
g.add(invtf.layers.Normalize()) 

# Actual model. 
g.add(invtf.layers.Squeeze())

for i in range(5): 
	g.add(invtf.layers.ActNorm())
	g.add(invtf.layers.Conv3DCirc())
	g.add(invtf.layers.AdditiveCouplingReLU(sign=+1))  

	g.add(invtf.layers.ActNorm())
	g.add(invtf.layers.Conv3DCirc())
	g.add(invtf.layers.AdditiveCouplingReLU(sign=-1))  
	
	if i == 3: g.add(invtf.layers.MultiScale()) 


# Prepare model for training and print summary. 
g.compile()  
g.init(X[:1000])  
g.summary()

#g.check(X)
g.print_loss(X[:100])

# Train model. 
g.fit(X, batch_size=512)

g.print_loss(X[:100])

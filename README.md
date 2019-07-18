*cite us using <a href="">bibtex</a>*
# InvTF
Train invertible generative models using the simplicity of Keras (aka normalizing flows).

<b>Example:</b>

```
import invtf
import tensorflow.keras as keras

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

for i in range(10): 
	g.add(invtf.layers.ActNorm())
	g.add(invtf.layers.Conv3DCirc())
	g.add(invtf.layers.AdditiveCouplingReLU()) 
	
	if i == 5: g.add(invtf.layers.MultiScale())

# Prepare model for training and print summary. 
g.compile()  
g.init(X[:1000])  
g.summary()

# Train model. 
g.fit(X, batch_size=512)
```

<img src="faces.png">

Most recent invertible generative model [1,2,3,4] have been <a href="">reproduced</a> in InvTF. Pretrained models are automatically downloaded when needed.

<b>Example</b>: Use pretrained model.

```
from invtf import Glow

glow.interpolate(faces()[0], faces()[1])

glow.generate(10)
```

<img src="interpolate.png">
<img src="generated.png">

Please see our <a href="">tutorial</a> and <a href="">documentation</a> for more information. 

TLDR: Easily train reversible generative models with Keras.

# Details

"invtf/"	the package directory. 

"articles/":	 notes / ipython notebook explaining previous articles reproducing their experiments in ipython notebooks.

"examples": 	implementation of a lot of examples, naming conforms with "dataset" x "model".py
		examples:
		"syntheticNormal_feedforward.py"
		"cifar10_glow.py"
		Besides show-casing how the framework can be used, these files are also used for test casing, just run 'run_all.py' (might take itme). 


# Development details

Developed using tensorflow-gpu 2.0beta, ubuntu 16.04 LTS, cuda 10, RTX 2080 8 GB



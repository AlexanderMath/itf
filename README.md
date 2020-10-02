# InvTF
Open-source version of internal framework used in research project. The following features have known bugs, use them at own risk! 
- O(1) or O(sqrt(L)) memory backpropagation.
- Variational dequantization. 
The animation below are produced using 
```
reproduce.py --problem celeb --model realnvp
reproduce.py --problem cifar --model realnvp
```
<div style="float: left; ">
<img src="animations/celeb.gif" style="width: 40%">
<img src="animations/cifar10.gif" style="width: 40%">
</div>

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



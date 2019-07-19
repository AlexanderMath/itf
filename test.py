"""
	Make a service that automatically runs all test cases every time code base changes. 
	If something breaks this should change a image on the README file.  
	Maybe someone already made such a thing, ask Lukas. 

	Current type of test cases: 

		- Shape computations with Squeeze and MultiScale architecture. 
		- Likelihood Computation (mainly Jacobian):
			- Layer-wise test of Jacobian computations (and combined model tests also). 
			- Optimality test: on toy data we can compute optimal model, and the likelihood
			  of this model can be computed which we can then check. (e.g. KL on normals)
		- Convergence tests on toy data sets. 
		

	Things that are useful during development, but won't be written explicitly into tests: 

		- Likelihood computations: 
			- Measure discrepancy between slow&exact likelihood computations and that 
			  computed using the def log_det() funtions. (by layer type and entire model). 
			- Investigate if encodings of data is "normally distributed". 

		- Invertibility: 
			- Measure inveribility error of entire model and each component. 
			  (by layer type and entire model) 

			

	TODO:
		A lot of the tests uses randomly initialized stuff. 
		Make sure numpy, tensorflow are all seeded and use fixed randomness to ensure 
		deterministic behaviour. 

		Remove a lot of the nasty printing information when doing unit tests. 
		It keeps printing: 

			WARNING: Logging before flag parsing goes to stderr.
			W0714 14:35:05.503682 140430992713472 deprecation.py:323] From /home/maximin/miniconda3/envs/tf/lib/python3.7/site-packages/tensorflow/python/ops/math_grad.py:1205: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
			Instructions for updating:
			Use tf.where in 2.0, which has the same broadcast rule as np.where

		Caused by tensorflow code, not sure how to make it be silent. 


	Potential other tests:
	
		Use other flow models and see how likely they deem data produced by models trained here. 
		The likelihood might not be meaningful, however, it might show potential bugs if there are 
		very big differences. 

"""
import unittest
#from test.shape 		import *
# from test.jacobian 		import *
#from test.optimality 	import *
# from test.inverse 		import *
from test.gradients import *
if __name__ == "__main__": 

	unittest.main()

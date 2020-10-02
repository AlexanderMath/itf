<div style="width: 100%; text-align: left; ">
<img src="animations/celeb.gif" width="45%">
<img src="animations/cifar10.gif" width="45%">
</div>

# InvTF
Open-source version of internal framework used in research project. The following features have known bugs, use them at own risk! 
- O(1) or O(sqrt(L)) memory backpropagation.
- Variational dequantization. 
The animation above were produced by runnign:  
```
reproduce.py --problem celeb --model realnvp
reproduce.py --problem cifar --model realnvp
```

# Development details

Developed using tensorflow-gpu 2.0beta, ubuntu 16.04 LTS, cuda 10, RTX 2080 8 GB



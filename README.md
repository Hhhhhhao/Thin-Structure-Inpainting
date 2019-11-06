# Adversarial-Gaps-Recovery-with-Policy-Gradient

Code for paper "Adversarial Large-scale Root Gaps Inpainting" and paper "Recovery of Gaps in Binary Segmentation Masks With Thin Structures With Adversarial Learning and Policy Gradient"

Authors: Hao Chen, Mario Valerio Giuffrida, Peter Doerner, and Sotirios A. Tsaftaris

***

## Content
* Installation
- Dataset
* Running

## Installation
We use Pytorch for training the models and our experiments. 
First create a new conda environment, then clone the repo by running to your directory:


`git clone git@git.ecdf.ed.ac.uk:s1786991/adversarial-gaps-recovery-with-policy-gradient.git`

Finally install the required packages:

`pip install -r requirements.txt`

## Dataset
Our project is running and testing on four datasets:
* Chickpea Root and [Synthetic Root](https://zenodo.org/record/61739) Segmentation
- [Satellite Road Segmentation](https://www.cs.toronto.edu/~vmnih/data/)
- [Retinal Vessel Segmentation](https://www5.cs.fau.de/research/data/fundus-images/)
* [Sketchy Database](http://sketchy.eye.gatech.edu/)

The chickpea root data is currently private and unavailable, but one can train the model only on synthetic root and apply the model on the desired root dataset.

Downloaded the dataset and put it in **data/dataset_name/** directory and split then into **train/valid/test** directories.

## Running
The specified dataset which the model will be trained on and all the hyper-parameters the model required is specified in configuration files in the **configs** directory.

To run the code and train the model (e.g. on root dataset):

`python train.py -c configs/root_config.json`

The trained model will be saved in **saved/** directory
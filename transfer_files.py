import os
import random
from shutil import copyfile
import numpy as np


def split_data(source,training_dir,valid_dir,num,split_fraction):

  l1 = len(os.listdir(source))
  rand_num1 = random.sample(range(0,l1,1),int(split_fraction*num))
  rand_num2 = random.sample([i for i in range(0,l1,1) if i not in rand_num1],int((1-split_fraction)*num))
  
  try:
      [copyfile(source+os.listdir(source)[i],training_dir+os.listdir(source)[i])
       for i in rand_num1 if os.path.getsize(source + os.listdir(source)[i])!=0]
    
      [copyfile(source+os.listdir(source)[i],valid_dir+os.listdir(source)[i])
       for i in rand_num2 if os.path.getsize(source+ os.listdir(source)[i])!=0]

  except FileNotFoundError:
    pass


  return


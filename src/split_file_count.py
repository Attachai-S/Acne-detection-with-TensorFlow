import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

'''
install library
Pip install pandasPip install pandas

'''
'''input your data path'''
train = r'C:\Users\vangu\PycharmProjects\Acne-detection-wth-TensorFlow\src\Acne04-Detection-5\test'
test = r'C:\Users\vangu\PycharmProjects\Acne-detection-wth-TensorFlow\src\Acne04-Detection-5\train'
valid = r'C:\Users\vangu\PycharmProjects\Acne-detection-wth-TensorFlow\src\Acne04-Detection-5\valid'

train_count = int(len([name for name in os.listdir(test) if os.path.isfile(os.path.join(test, name))]) - 1) #excep _annotation.csv
test_count = int(len([name for name in os.listdir(train) if os.path.isfile(os.path.join(train, name))]) - 1)
valid_count = int(len([name for name in os.listdir(valid) if os.path.isfile(os.path.join(valid, name))]) - 1)
sum = train_count + test_count + valid_count

print(f'Total number of train files: {train_count} , as {round((train_count / sum)*100,2)} percent of data')
print(f'Total number of test files: {test_count} , as {round((test_count / sum)*100,2)} percent of data')
print(f'Total number of valid files: {valid_count} , as {round((valid_count / sum)*100,2)} percent of data')

print(f"\n{"="*100}\n")


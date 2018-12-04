# MNIST dataset is Hello World of Classification Problems
#       which is a set of 70,000 small images of digits handwritten
#               by high school students and employees of the US Census Bureau

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
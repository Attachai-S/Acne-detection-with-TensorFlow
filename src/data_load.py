# for jupiter notebook
# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="CvCELtVsjgcYQvHFl03o")
project = rf.workspace("acne-vulgaris-detection").project("acne04-detection")
version = project.version(5)
dataset = version.download("tensorflow")

# -*- coding: utf-8 -*-
"""styleClassificationFastaiCSV.ipynb

Automatically generated by Colaboratory.

Original file is located at
  https://colab.research.google.com/drive/1XR7_7xGULam4mq-BwlT_MzmaWieLbC-s

# CS121  and CS152 - Assignment 2A

In this activity you'll download two classes of images, train a
classifier to distinguish them, and analyze the results.

##  Upload your images

Use Google images to download two classes of images into separate
folders on your computer. Zip each folder.

In the file view of your paperspace machine, navigate to the data
directory. Create a new folder called 'AB' where A is the name of your
first class and B is the name of your second class.

Run the following code to do some setup. Replace A and B with your
category names.

"""

from fastai.vision import *
import pathlib
import PIL
import pandas as pd
import matplotlib.pyplot as plt
import numpy
import warnings

# Prevent warnings from appearing more than once
warnings.filterwarnings('ignore')


# Create a textfile holding our output
out = open("style_unfreeze_300_output.txt", "w")


## Read the cleaned CSVs and create a df
root = "wikiart/"
train = pd.read_csv(root+"wikiart_csv/style_train_clean.csv", header=None, index_col=0)
valid = pd.read_csv(root+"wikiart_csv/style_valid_clean.csv", header=None, index_col=0)
df = pd.concat([train, valid])
out.write("Styles Dataframe\n")
out.writelines([df.head().to_string(), "\n", "\n"])

style_dict = {0: "Abstract_Expressionism", 1: "Action_painting", 2: "Analytical_Cubism",
              3: "Art_Nouveau", 4: "Baroque", 5: "Color_Field_Painting",
              6: "Contemporary_Realism", 7: "Cubism", 8: "Early_Renaissance",
              9: "Expressionism", 10: "Fauvism", 11: "High_Renaissance",
              12: "Impressionism", 13: "Mannerism_Late_Renaissance",
              14: "Minimalism", 15: "Naive_Art_Primitivism", 16: "New_Realism",
              17: "Northern_Renaissance", 18: "Pointillism", 19: "Pop_Art",
              20: "Post_Impressionism", 21: "Realism", 22: "Rococo",
              23: "Romanticism", 24: "Symbolism", 25: "Synthetic_Cubism",
              26: "Ukiyo_e" }


## Sample max 200 photos/category
style_list = []
num_sample = 300
df.columns = ["path", "label"]

for key in style_dict:
  temp = df[df.label == style_dict[key]]
  if (len(temp.index) >= num_sample):
    temp = temp.sample(n = num_sample, random_state=1)
  style_list.append(temp)

style_df = pd.concat(style_list)
out.write("Sampled Styles Dataframe\n")
out.writelines([style_df.head().to_string(), "\n", "\n"])


## Create an ImageDataBunch from the dataframe
styles_path = root+"wikiart"
data = ImageDataBunch.from_df(df=style_df, path=styles_path,
                              valid_pct=0.2,
                              ds_tfms=get_transforms(),
                              size=180, num_workers=0).normalize(imagenet_stats)


## Verify the number of images
out.write("Verify number of images per class\n")
out.writelines(data.classes)
out.write("\n")
out.writelines(["Training set size: ", str(len(data.train_ds)), "\n",
                "Validation set size: ", str(len(data.valid_ds))])


## Create a CNN learner and train
learn = create_cnn(data, models.resnet34, metrics=error_rate)

"""
## Find a better learning rate - Optimal LR: 1e-01
learn.lr_find()
learn.recorder.plot()
plt.show()
plt.savefig("style_lr_300.png")
"""


## Train with custom learning rate
learn.unfreeze()
learn.fit_one_cycle(20, max_lr=slice(1e-4, 1e-2))


## Export the learner
learn.save("style_unfreeze_300")
learn.export("./style_unfreeze_300.pkl")


## Show and save top losses
learn.load("style_unfreeze_300")
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
plt.savefig("style_unfreeze_300_top_losses.png")


## Show and save confusion matrix
interp.plot_confusion_matrix()
plt.savefig("style_unfreeze_300_confusion_matrix.png")
confused = interp.most_confused(min_val=2)
for style_class in confused:
  out.writelines(str(style_class))
  out.write("\n")


# Close the output file
out.close()

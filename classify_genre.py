# CS121 - Genre Model Training

##  Using data from ArtGAN Github and after running cleanGenreCsv.py to get a cleaned version of the ArtGAN genre CSV


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
out = open("genre_unfreeze_300_output.txt", "w")


## Read the cleaned CSVs and create a df
root = "wikiart/"
train = pd.read_csv(root+"wikiart_csv/genre_train_clean.csv", header=None, index_col=0)
valid = pd.read_csv(root+"wikiart_csv/genre_valid_clean.csv", header=None, index_col=0)
df = pd.concat([train, valid])
out.write("Genre Dataframe\n")
out.writelines([df.head().to_string(), "\n", "\n"])

genre_dict = {0: "Abstract", 1:"Cityscape", 2:"Genre Art", 3:"Illustration", 
              4:"Landscape", 5:"Nude Painting", 6:"Portrait", 7:"Religious Painting", 
              8:"Sketch and Study", 9:"Still Life"}


## Sample max 300 photos/category
genre_list = []
num_sample = 300
df.columns = ["path", "label"]

for key in genre_dict:
  temp = df[df.label == genre_dict[key]]
  if (len(temp.index) >= num_sample):
    temp = temp.sample(n = num_sample, random_state=1)
  genre_list.append(temp)

genre_df = pd.concat(genre_list)
out.write("Sampled Genre Dataframe\n")
out.writelines([genre_df.head().to_string(), "\n", "\n"])


## Create an ImageDataBunch from the dataframe
genre_path = root+"wikiart"
data = ImageDataBunch.from_df(df=genre_df, path=genre_path,
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
## Find a better learning rate - Optimal LR: 1e-02
learn.lr_find()
learn.recorder.plot()
plt.show()
plt.savefig("genre_lr_300.png")
"""


## Train with custom learning rate
learn.unfreeze()
learn.fit_one_cycle(20, max_lr=slice(1e-4, 1e-2))


## Export the learner
learn.save("genre_unfreeze_300")
learn.export("./genre_unfreeze_300.pkl")


## Show and save top losses
learn.load("genre_unfreeze_300")
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
plt.savefig("genre_unfreeze_300_top_losses.png")


## Show and save confusion matrix
interp.plot_confusion_matrix()
plt.savefig("genre_unfreeze_300_confusion_matrix.png")
confused = interp.most_confused(min_val=2)
for genre_class in confused:
  out.writelines(str(genre_class))
  out.write("\n")


# Close the output file
out.close()

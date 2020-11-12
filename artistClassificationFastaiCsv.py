# CS121 - Artist Model Training

##  Using data from ArtGAN Github and after running cleanArtistCsv.py to get a cleaned version of the ArtGAN artist CSV


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
out = open("artist_unfreeze_300_output.txt", "w")


## Read the cleaned CSVs and create a df
root = "wikiart/"
train = pd.read_csv(root+"wikiart_csv/artist_train_clean.csv", header=None, index_col=0)
valid = pd.read_csv(root+"wikiart_csv/artist_valid_clean.csv", header=None, index_col=0)
df = pd.concat([train, valid])
out.write("Artist Dataframe\n")
out.writelines([df.head().to_string(), "\n", "\n"])

artist_dict = {0: "Albrecht Durer", 1:"Boris Kustodiev", 2:"Camille Pissarro", 3:"Childe Hassam", 4:"Claude Monet", 5:"Edgar Degas", 
              6:"Eugene Boudin", 7:"Gustave Dore", 8:"Ilya Repin", 9:"Ivan Aivazovsky", 10:"Ivan Shishkin", 11: "John Singer Sargent",
              12: "Marc Chagall", 13: "Martiros Saryan", 14: "Nicholas Roerich", 15: "Pablo Picasso", 16:"Pablo Cezanne", 17:"Pierre Auguste Penoir",
              18: "Pyotr Konchalovsky", 19:"Raphael Kirchner", 20: "Rembrandt", 21:"Salvador Dali", 22:"Vincent van Gogh"} 


## Sample max 200 photos/category
artist_list = []
num_sample = 200
df.columns = ["path", "label"]

for key in artist_dict:
  temp = df[df.label == artist_dict[key]]
  if (len(temp.index) >= num_sample):
    temp = temp.sample(n = num_sample, random_state=1)
  artist_list.append(temp)

artist_df = pd.concat(artist_list)
out.write("Sampled Artist Dataframe\n")
out.writelines([artist_df.head().to_string(), "\n", "\n"])


## Create an ImageDataBunch from the dataframe
artist_path = root+"wikiart"
data = ImageDataBunch.from_df(df=artist_df, path=artist_path,
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
plt.savefig("artist_lr_300.png")
"""


## Train with custom learning rate
learn.unfreeze()
learn.fit_one_cycle(20, max_lr=slice(1e-4, 1e-2))


## Export the learner
learn.save("artist_unfreeze_300")
learn.export("./artist_unfreeze_300.pkl")


## Show and save top losses
learn.load("artist_unfreeze_300")
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
plt.savefig("artist_unfreeze_300_top_losses.png")


## Show and save confusion matrix
interp.plot_confusion_matrix()
plt.savefig("artist_unfreeze_300_confusion_matrix.png")
confused = interp.most_confused(min_val=2)
for artist_class in confused:
  out.writelines(str(artist_class))
  out.write("\n")


# Close the output file
out.close()

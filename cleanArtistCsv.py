from collections import namedtuple
import pathlib
import PIL
import pandas as pd

# Read CSVs
root = "wikiart/"
train = pd.read_csv(root+"wikiart_csv/artist_train.csv", header=None)
valid = pd.read_csv(root+"wikiart_csv/artist_val.csv", header=None)

artist_dict = {0: "Albrecht Durer", 1:"Boris Kustodiev", 2:"Camille Pissarro", 3:"Childe Hassam", 4:"Claude Monet", 5:"Edgar Degas", 
              6:"Eugene Boudin", 7:"Gustave Dore", 8:"Ilya Repin", 9:"Ivan Aivazovsky", 10:"Ivan Shishkin", 11: "John Singer Sargent",
              12: "Marc Chagall", 13: "Martiros Saryan", 14: "Nicholas Roerich", 15: "Pablo Picasso", 16:"Pablo Cezanne", 17:"Pierre Auguste Penoir",
              18: "Pyotr Konchalovsky", 19:"Raphael Kirchner", 20: "Rembrandt", 21:"Salvador Dali", 22:"Vincent van Gogh"} 

## Create a namedtuple
Pandas = namedtuple("Pandas", "index pathname classname")

## Trained CSV
# Iterate over rows and add only if path exists
train_cleaned = []
for row in train.itertuples():
  if (pathlib.Path(root+"wikiart/"+row[1]).exists()):
    classed = Pandas(row[0], row[1], artist_dict[row[2]])
    train_cleaned.append(classed)

# Write to CSV
trained_clean_df = pd.DataFrame(train_cleaned)
trained_clean_df.to_csv(path_or_buf=root+"wikiart_csv/artist_train_clean.csv",
                        index=None, header=None)

## Valid CSV
# Iterate over rows and add only if path exists
valid_cleaned = []
for row in valid.itertuples():
  if (pathlib.Path(root+"wikiart/"+row[1]).exists()):
    classed = Pandas(row[0], row[1], artist_dict[row[2]])
    valid_cleaned.append(classed)

# Write to CSV
valid_clean_df = pd.DataFrame(valid_cleaned)
valid_clean_df.to_csv(path_or_buf=root+"wikiart_csv/artist_valid_clean.csv",
                      index=None, header=None)

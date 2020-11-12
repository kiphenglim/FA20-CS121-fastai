from collections import namedtuple
import pathlib
import PIL
import pandas as pd

# Read CSVs
root = "wikiart/"
train = pd.read_csv(root+"wikiart_csv/genre_train.csv", header=None)
valid = pd.read_csv(root+"wikiart_csv/genre_val.csv", header=None)

genre_dict = {0: "Abstract", 1:"Cityscape", 2:"Genre Art", 3:"Illustration", 
              4:"Landscape", 5:"Nude Painting", 6:"Portrait", 7:"Religious Painting", 
              8:"Sketch and Study", 9:"Still Life"}

## Create a namedtuple
Pandas = namedtuple("Pandas", "index pathname classname")

## Trained CSV
# Iterate over rows and add only if path exists
train_cleaned = []
for row in train.itertuples():
  if (pathlib.Path(root+"wikiart/"+row[1]).exists()):
    classed = Pandas(row[0], row[1], genre_dict[row[2]])
    train_cleaned.append(classed)

# Write to CSV
trained_clean_df = pd.DataFrame(train_cleaned)
trained_clean_df.to_csv(path_or_buf=root+"wikiart_csv/genre_train_clean.csv",
                        index=None, header=None)

## Valid CSV
# Iterate over rows and add only if path exists
valid_cleaned = []
for row in valid.itertuples():
  if (pathlib.Path(root+"wikiart/"+row[1]).exists()):
    classed = Pandas(row[0], row[1], genre_dict[row[2]])
    valid_cleaned.append(classed)

# Write to CSV
valid_clean_df = pd.DataFrame(valid_cleaned)
valid_clean_df.to_csv(path_or_buf=root+"wikiart_csv/genre_valid_clean.csv",
                      index=None, header=None)

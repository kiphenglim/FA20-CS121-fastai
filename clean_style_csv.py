""" You can download the Style CSVs from the cs-chan/ArtGAN Github
repository. Assuming that you have already downloaded the entire ~28
GB dataset from the provided link, this script removes rows from the
Style CSV if that image is not present on your local computer. """

from collections import namedtuple
import pathlib
import pandas as pd

# Read CSVs
ROOT = "wikiart/"
train = pd.read_csv(ROOT+"wikiart_csv/style_train.csv", header=None)
valid = pd.read_csv(ROOT+"wikiart_csv/style_val.csv", header=None)

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

## Create a namedtuple
Pandas = namedtuple("Pandas", "index pathname classname")

## Trained CSV
# Iterate over rows and add only if path exists
train_cleaned = []
for row in train.itertuples():
    if pathlib.Path(ROOT+"wikiart/"+row[1]).exists():
        classed = Pandas(row[0], row[1], style_dict[row[2]])
        train_cleaned.append(classed)

# Write to CSV
trained_clean_df = pd.DataFrame(train_cleaned)
trained_clean_df.to_csv(path_or_buf=ROOT+"wikiart_csv/style_train_clean.csv",
                        index=None, header=None)

## Valid CSV
# Iterate over rows and add only if path exists
valid_cleaned = []
for row in valid.itertuples():
    if pathlib.Path(ROOT+"wikiart/"+row[1]).exists():
        classed = Pandas(row[0], row[1], style_dict[row[2]])
        valid_cleaned.append(classed)

# Write to CSV
valid_clean_df = pd.DataFrame(valid_cleaned)
valid_clean_df.to_csv(path_or_buf=ROOT+"wikiart_csv/style_valid_clean.csv",
                      index=None, header=None)

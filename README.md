# FA20-CS121-fastai

Harvey Mudd College FA20 CS121 Section 2: Deep Learning Classification
of Art Pieces

## About

This repository contains the fast.ai models and training output that
is used in our [web app](https://github.com/kiphenglim/FA20-CS121).


## Source Data

We trained using data from the
[cs-chan/ArtGAN](https://github.com/cs-chan/ArtGAN) dataset. The
ArtGAN ReadMe links to a site where a large photo dataset can be
downloaded. Within the ArtGAN Github itself, there is are CSV files
for artist, genre, and style which you will also need.


## Usage

Once you have these things on your local computer, you can run our
three cleaning scripts: clean_artist_csv.py, clean_genre_csv.py, and
clean_style_csv.py. Each of these scripts will produce a new CSV where
the entries in the CSV only contain rows that reference photos that
are obtained from the photo download mentioned above. We found some
inconsistencies between the CSV and photo downloads that neccessitated
this extra step.

After this, you can run classify_artist.py, classify_genre.py, and
classify_style.py using fast.ai. Each of these scripts will output a
model, a confusion matrix, and a photo of the top losses. The top
losses and confusion matrix are both saved as images in the local
directory. Additionally, the top losses is saved as a list inside a
console.txt file along with some other useful output for
debugging. These models can then be used to predict the artist, genre,
and style of paintings.

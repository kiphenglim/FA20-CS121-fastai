# FA20-CS121-fastai

Harvey Mudd College FA20 CS121 Section 2: Deep Learning Classification of Art Pieces

This repository contains the fastai models and training output for our project. We used data from the ArtGAN Github account, which can be found 
at the following link: https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset. The ArtGAN ReadMe links to a site where a large photo dataset can be downloaded. Within the ArtGAN Github itself within the ArtGAN/WikiArt Dataset there are CSV files for artist, genre, and style which you will also need. 

Once you have these things on your local computer, you can run our three cleaning scripts: cleanStyleCsv.py, cleanGenreCsv.py, and cleanArtistCsv.py. Each of these scripts will produce a new CSV where the entries in the CSV only contain links to photos that are obtained from the photo download mentioned above. We found some inconsistencies between the CSV and photo downloads that neccessitated this extra step.

After this, you can run styleClassificationFastaiCSV.py, genreClassificationFastaiCSV.py, and artistClassificationFastaiCSV.py using Fast.ai. Each of these scripts will output a model, a confusion matrix, and a photo of the top losses. These models can then be used to predict the artist, genre, and style of paintings. They are used in our art classification app, which can be found at the following Github: https://github.com/kiphenglim/FA20-CS121.
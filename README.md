This repository was developed while I was interning in the Data Science Lab at the Smithsonian institution. The repository is forked to the SI data science GitHub at https://github.com/sidatasciencelab/pix_plot_gooey where it lives.

# pix_plot_gooey
A Gooey version of Pix Plot (https://github.com/YaleDHLab/pix-plot). 
Pix_plot_gooey is a tool that provides users with the ability to visualize and analyze thousands of images in a two-dimensional projection by comparing and clustering the images. The image analysis uses Tensorflow's Inception bindings, and the visualization layer uses a custom WebGL viewer.

## Setup
To clone this repository to you computer use 

```
git clone https://github.com/sethbwilson/pix_plot_gooey.git && cd pix_plot_gooey
```
Then, to create a new envitonment with all the correct dependencies run:
```
conda env create -f environment.yml
```
Activate that environment by running:
```
conda activate pix
```

## Running
Launch code from terminal using command

```
pythonw pix_plot_gooey.py
```

A window will open up with several spaces to fill in. 

![alt text](https://github.com/sethbwilson/pix_plot_gooey/blob/master/Gooey%20interface.png)

The fields are described below.

|Field Name | Field Description|
|---|---|
| image_dir | The folder containing all of your images. Click the browse button and use the finder screen to select and open the folder that contains your images. |
| model_use | The file containing the model you plan to use. Click the browse button and use the finder screen to select and open the model file. This should be found in the models folder built in the pix_plot_gooey folder.
| Clusters | Provide a number that the program will use to find hotspots in your data. The number provided will be the umber of hotspots the program will find. Pick a number that is close the the estimated nmber of categories you think are in the image set. The number provided must be less than the number of images in the data set. This is purely a browsing feature, it does not change the projection.|
| output | This is the folder where all the information generated form the program will be stored. It is important that this folder is found within the pix-plot folder that the program generated upon launching. Click on the brose button, fing the pix-plot folder, double click on the folder and then make a new folder called output by clicking the new folder button. Open that folder for the program. 
| method | This is a more technical feature. Select UMAP. |

## Launching Viewer
After the processing is finished, open a Google Chrome window and enter localhost:8000 in the search bar. The results of the processed are viewed in the web browser.

## Next Steps
- Installable on Windows and Mac
- Changes to HDBSCAN to find optimal number of clusters
- Search feature
- Detail feature

## Acknowledgements
This project was made possible by the Smithsonian Data Science Lab in the OCIO and the Yale Digital Humanities Lab. 

# Shape Modeling and Geometry Processing - Final project <br>
- Alessia Paccagnella
- Yingyan Xu
- Xiaojing Xia
- Niklaus Houska
- Alexandre Binninger

## How to select landmarks

```
1. Press S to go into Select Mode
2. Click to select landmarks (vertices)
3. Press A to apply landmark selection (or else selection will not be saved)
4. Click save landmarks to save landmarks to a file.

```

## How to run non-rigid alignment demo

- Click the `Rigid Alignment` button first
- Keep clicking `Non-Rigid Warping` until the result looks nice
- Click `Display Template` to see the warped template
- Tik `high res` option to use high resolution template (The default is low resolution.)
- If you change the template resolution or grid resolution, start again from rigid alignment
- `threshold` is now gradually increasing (change `INC` to set the increment per iteration), you can manually set `threshold` in each warping iteration

*Note that when generating the aligned meshes, we use `INC = 0.1` and the termination condition is decided automatically.*


## Align all meshes

- smoothed meshes and landmark files in `data/smoothed`
- mesh namelist saved in `data/smoothed/smoothed_mesh_list` (we splitted the meshes to be smoothed between groups. Each group had 12-13 meshes to be smoothed and their landmarks to be marked. However, group 12 did not smooth their meshes, so our group did it also for them. Group 1 did not contribute to the work.)
- click `Align All Meshes` to process all meshes and save alligned meshes in `data/aligned`

## PCA face

### GUI, save and loads.

- To run the PCA, you can click on the button *Files to run SVD on* and select a file in a folder named `list_entry_files/`. For instance, `data/aligned_faces_example/example1/list_entry_files/list_entry_files_9.txt`.
- The button `Run SVD` will run the SVD on the selected files. To select the number of eigenfaces, you have to tune the parameter `nb_eigenfaces` directly in main.cpp and recompile the code. (Some variables depend on the number of eigenfaces, this is why this parameter is not easy to tune directly in UI).
- After Running SVD, you can save it by clicking on `Save Results`. It will create (or overwrite) a folder `results_eigenfaces` directly in `build`.
- You can load the results of an SVD. To that, you have to click on `Load SVD Results` and choose a file which contains the path to a result folder. For instance, `data/results_eigenfaces/example1/folder.txt`. 

### How does it work?

The default number of eigenfaces is 8. After running SVD or directly loading the results, the mean face appears (the weights of eigenfaces is therefore 0). You can either modify the weights directly in the UI or select 2 faces in the Morphing section and blend between both of them. 


## Repartition of work.

### Alessia Paccagnella
- Leader of the group, coordinated team and deadlines for the different steps.
- Worked on the preprocessing Laplacian smoothing step for our data. After smoothing our group's meshes, I marked and saved the landmarks. 

### Yingyan Xu

- Rigid Alignment.
- Non-rigid warping.
- Generate aligned meshes.

### Xiaojing Xia

- Coordinated with the rest of the class to create a template for marking landmarks and splitting up the data for preprocessing among groups. 
- Worked on the UI to mark and save landmarks.
- Helped write framework of Non-rigid warping and generate aligned meshes.

### Alexandre Binninger

- PCA on faces.
- Some UI for eigenfaces.
- Saving and Loading system for PCA results.

### Niklaus Houska

- Eigen face user interaction, morphing and exporting

N.B.: We all worked on the presentation slides and on the report.


## Results

### Smoothed meshes

<img src="https://github.com/alessiapacca/igl-project/blob/master/igl-final-project/results/smoothed.png" width="200"/>

### Mark Landmarks

<img src="https://github.com/alessiapacca/igl-project/blob/master/igl-final-project/results/mark_landmarks.png" width="200"/> 

### Aligned (Rigid and Non rigid)

<img src="https://github.com/alessiapacca/igl-project/blob/master/igl-final-project/results/rigid.png" width="200"/> <img src="https://github.com/alessiapacca/igl-project/blob/master/igl-final-project/results/aligned.png" width="200"/> 

### PCA

#### Mean Face
<img src="https://github.com/alessiapacca/igl-project/blob/master/igl-final-project/results/mean_face.png" width="200"/>

#### Example of a created smiling Face

<img src="https://github.com/alessiapacca/igl-project/blob/master/igl-final-project/results/example_pca_face.png" width="200"/>

#### Face Morphing

<img src="https://github.com/alessiapacca/igl-project/blob/master/igl-final-project/results/face_morphing.gif" alt="animated">


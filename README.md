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

```

---

## rigid alignment example

- Currently using only the provided landmark example
- To be updated when more landmark data available
- Click the `Rigid Alignment` button to see the demo

---

## how to run non-rigid alignment example

- Click the `Rigid Alignment` button first
- Click the `Non-Rigid Warping` until the number of closest point constraint converges
- Click the `Display Template` to see the warping result
- If you reset threshold or resolution, start again from rigid alignment

---

## alignment update

- smoothed meshes and landmark files in `data/smoothed`
- mesh namelist saved in `data/smoothed/smoothed_mesh_list` (not include group12)
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

### Yingyan Xu

### Xiaojing Xia

### Alexandre Binninger

Running PCA on faces, some UI. Saving and Loading system for PCA results.

### Niklaus Houska


N.B.: We all worked on the presentation slides and on the report.


## Results

### Smoothed meshes

<img src="https://github.com/alessiapacca/igl-project/igl-final-project/results/smoothed.png" width="400"/>

### Mark Landmarks

<img src="https://github.com/alessiapacca/igl-project/igl-final-project/results/mark_landmarks.png" width="400"/>

### Aligned (Rigid and Non rigid)

<img src="https://github.com/alessiapacca/igl-project/igl-final-project/results/aligned.png" width="400"/>

### PCA

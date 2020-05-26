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

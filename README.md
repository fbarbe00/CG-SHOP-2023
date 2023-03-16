# DACS-Project-1-1
Repository to solve the Minimum Coverage by Convex Polygons given by CG:SHOP 2023.

## Setting up this repository
After cloning this repository, all necessary dependencies can be installed by running:
```bash
pip install -r requirements.txt
```
Python >3.10 is required to run this code.

The `redumis` executable needs to be placed in the `tools` folder. It can be found in the [KaMIS repository](https://github.com/KarlsruheMIS/KaMIS).

## Running the code
Most files can take a `-h` argument and return a detailed description of the arguments they take.

- `find_invisible_points.py`: given a file with a polygon, saves the list of points that are not visible from the polygon in a file.
- `find_coverage.py`: given a file with a polygon and a list of points, saves a coverage with convex polygons in a file, removing overlapping polygons.

## References
This code uses Triangle (https://rufat.be/triangle/) to triangulate the convex polygons, and KaMIS (https://github.com/KarlsruheMIS/KaMIS) to find the Maximum Independent Set of a graph.

## Links
https://cgshop.ibr.cs.tu-bs.de/competition/cg-shop-2023/

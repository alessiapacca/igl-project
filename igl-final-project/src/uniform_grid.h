#ifndef __Uniform_Grid__
#define __Uniform_Grid__

#include <unordered_map>

using namespace std;
using namespace Eigen;

class UniformGrid
{
public:
    int xres = 20, yres = 20, zres = 20; // resolution
    RowVector3d bb_min, bb_max, dim; // bounding box
    double enlargeCoeff = 1.1; // enlargement factor
    double dx, dy, dz; // grid spacing
    double offset_x, offset_y, offset_z; // offset after enlarging bounding box

    UniformGrid() {}

    UniformGrid(const RowVector3d& bb_min, const RowVector3d& bb_max,
                const int xres, const int yres, const int zres) 
                : bb_min(bb_min), bb_max(bb_max), 
                xres(xres), yres(yres), zres(zres) {

        this->dim = bb_max - bb_min;

        // slightly enlarge bounding box
        this->dx = enlargeCoeff * dim[0] / (double)(xres - 1);
        this->dy = enlargeCoeff * dim[1] / (double)(yres - 1);
        this->dz = enlargeCoeff * dim[2] / (double)(zres - 1);

        this->offset_x = 0.5 * (enlargeCoeff - 1) * dim[0];
        this->offset_y = 0.5 * (enlargeCoeff - 1) * dim[1];
        this->offset_z = 0.5 * (enlargeCoeff - 1) * dim[2];

        // modify bb_min for enlargement
        this->bb_min -= RowVector3d(offset_x, offset_y, offset_z);

    }

    ~UniformGrid() {}

    // save grid index for each point of P
    void init_grid(const MatrixXd& P) {       
        p_in_grid.clear();
        for(int i=0; i<P.rows(); i++) {
            tuple<int, int, int> xyz = get_grid_index(P.row(i));
            p_in_grid[to_idx(xyz)].push_back(i);
        }
    }

    // find closest vertex of P in V, save the index and return the distance
    // only search in the y direction since it's orthogonal to the face plane
    // not really using it in the final code
    double query(const RowVector3d& P, const MatrixXd& V, int& index, const double threshold) {

        // grid index of the query point
        tuple<int, int, int> xyz = get_grid_index(P);

        // need to search at most 2 * diff grids
        int diff = abs(get<1>(xyz) - threshold / dy) + 1;
        
        // if points exist in closer grids, flag = true
        bool flag = false;

        double min_dist = __DBL_MAX__;
        for (int i=0; i<=diff && !flag; i++) {
            searched.clear();
            searched = vector<bool>(2 * xres * yres * zres, false);
            // front
            closest_in_grid(P, V, index, min_dist, flag, get<0>(xyz), get<1>(xyz) - i, get<2>(xyz));
            // back
            closest_in_grid(P, V, index, min_dist, flag, get<0>(xyz), get<1>(xyz) + i, get<2>(xyz));
        }

        if (min_dist == __DBL_MAX__) {
            return -1;
        } else {
            return min_dist;
        }       
    }

    // find closest vertex of P in V, save the index and return the distance
    // search all xyz directions
    double query_xyz(const RowVector3d& P, const MatrixXd& V, int& index, const double threshold) {
        
        // grid index of the query point
        tuple<int, int, int> xyz = get_grid_index(P);

        // the max number of grids we need to search along each direction
        int diff_x = abs(get<0>(xyz) - threshold / dx) + 1;
        int diff_y = abs(get<1>(xyz) - threshold / dy) + 1;
        int diff_z = abs(get<2>(xyz) - threshold / dz) + 1;
        int diff = max(max(diff_x, diff_y), diff_z);

        searched.clear();
        searched = vector<bool>(2 * xres * yres * zres, false);
        
        // for early break
        // terminate the search if vertices exist in inner "layers" of the grids
        bool flag = false; 
        
        int delta = 0;
        double min_dist = __DBL_MAX__;

        while (delta++ <= diff && !flag) {
            for (int x=get<0>(xyz)-delta; x<=get<0>(xyz)+delta; x++) {
                for (int y=get<1>(xyz)-delta; y<=get<1>(xyz)+delta; y++) {
                    for (int z=get<2>(xyz)-delta; z<=get<2>(xyz)+delta; z++) {
                        closest_in_grid(P, V, index, min_dist, flag, x, y, z);
                    }
                }
            }
        }

        if (min_dist == __DBL_MAX__) {
            return -1;
        } else {
            return min_dist;
        }
    }

private:

    // store all the vertex indices of V in each grid
    unordered_map<int, vector<int>> p_in_grid;

    // record whether a grid has been searched for the query vertex
    vector<bool> searched;

    inline int to_idx(const int x, const int y, const int z) {
        return x + xres * (y + yres * z);
    }

    inline int to_idx(const tuple<int, int, int> &xyz) {
        int x = get<0>(xyz);
        int y = get<1>(xyz);
        int z = get<2>(xyz);
        return x + xres * (y + yres * z);
    }

    // given point position, return which grid it's in
    inline tuple<int, int, int> get_grid_index(const RowVector3d &p) {
        RowVector3d diff = p - bb_min;
        int idx_x = floor((diff[0] + offset_x) / dx);
        int idx_y = floor((diff[1] + offset_y) / dy);
        int idx_z = floor((diff[2] + offset_z) / dz);

        return make_tuple(idx_x, idx_y, idx_z);
    }

    inline void closest_in_grid(const RowVector3d& P, const MatrixXd& V, int& index,
                                double& min_dist, bool& flag,
                                int x, int y, int z) {
        if (!isValid(x, y, z)) return;
        int idx = to_idx(x, y, z);
        if (searched[idx]) return;
        searched[idx] = true;
        for (int p : p_in_grid[idx]) {
            double cur_dist = (P - V.row(p)).norm();
            if (cur_dist < min_dist) {
                min_dist = cur_dist;
                index = p;
                flag = true;
            }
        }
    }

    // whether the given indices are legal, i.e., 0 <= idx < res
    inline bool isValid(int x, int y, int z) {
        return 0 <= x && x < xres && 0 <= y && y < yres && 0 <= z && z < zres;
    }
};

#endif

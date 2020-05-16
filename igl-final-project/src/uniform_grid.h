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

    // find closest point of P in V, save the index and return the distance
    double query(const RowVector3d& P, const MatrixXd& V, int& index, const double threshold) {

        // grid index of the query point
        tuple<int, int, int> xyz = get_grid_index(P);

        // need to search at most 2 * diff grids
        int diff = abs(get<1>(xyz) - threshold / dy) + 1;
        
        // if points exist in closer grids, flag = true
        bool flag = false;

        // only search in the y direction since it's orthogonal to the face plane
        int idx = 0, y = 0;
        double min_dist = __DBL_MAX__;
        for (int i=0; i<=diff && !flag; i++) {
            // front
            y = max(0, get<1>(xyz) - i);
            idx = to_idx(get<0>(xyz), y, get<2>(xyz));
            for (int p : p_in_grid[idx]) {
                double cur_dist = (P - V.row(p)).norm();
                if (cur_dist < min_dist) {
                    min_dist = cur_dist;
                    index = p;
                    flag = true;
                }
            }

            // back 
            if (i != 0) { // avoid repetition
                y = min(yres-1, get<1>(xyz) + i);
                idx = to_idx(get<0>(xyz), y, get<2>(xyz));
                for (int p : p_in_grid[idx]) {
                    double cur_dist = (P - V.row(p)).norm();
                    if (cur_dist < min_dist) {
                        min_dist = cur_dist;
                        index = p;
                        flag = true;
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

    // store all the point indices of V in each grid
    unordered_map<int, vector<int>> p_in_grid;

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
};

#endif

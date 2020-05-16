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

        this->bb_min -= RowVector3d(offset_x, offset_y, offset_z);

    }

    ~UniformGrid() {}

    void init_grid(const MatrixXd& P) {       
        p_in_grid.clear();
        for(int i=0; i<P.rows(); i++) {
            tuple<int, int, int> xyz = get_grid_index(P.row(i));
            p_in_grid[to_idx(xyz)].push_back(i);
        }
    }

    double query(const RowVector3d& P, const MatrixXd& V, int& index, const double threshold) {

        tuple<int, int, int> xyz = get_grid_index(P);

        // int diff_x = abs(get<0>(xyz) - threshold / dx) + 1;
        int diff_y = abs(get<1>(xyz) - threshold / dy) + 1;
        // int diff_z = abs(get<2>(xyz) - threshold / dz) + 1;

        double min_dist = __DBL_MAX__;
        // for (int x = max(0, get<0>(xyz) - diff_x); x <= min(xres-1, get<0>(xyz) + diff_x); x++) {
        //     for (int y = max(0, get<1>(xyz) - diff_y); y <= min(yres-1, get<1>(xyz) + diff_y); y++) {
        //         for (int z = max(0, get<2>(xyz) - diff_z); z <= min(zres-1, get<2>(xyz) + diff_z); z++) {
        //             int idx = to_idx(x, y, z);
        //             if(p_in_grid.count(idx) == 0) continue;
        //             for (int p : p_in_grid[idx]) {
        //                 RowVector3d P2 = V.row(p);
        //                 double cur_dist = (P - P2).norm();
        //                 if (cur_dist < min_dist) {
        //                     min_dist = cur_dist;
        //                     index = p;
        //                 }
        //             }
        //         }
        //     }
        // }
        for (int y = max(0, get<1>(xyz) - diff_y); y <= min(yres-1, get<1>(xyz) + diff_y); y++) {
            int idx = to_idx(get<0>(xyz), y, get<2>(xyz));
            if(p_in_grid.count(idx) == 0) continue;
            for (int p : p_in_grid[idx]) {
                RowVector3d P2 = V.row(p);
                double cur_dist = (P - P2).norm();
                if (cur_dist < min_dist) {
                    min_dist = cur_dist;
                    index = p;
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

    inline tuple<int, int, int> get_grid_index(const RowVector3d &p) {
        RowVector3d diff = p - bb_min;
        int idx_x = floor((diff[0] + offset_x) / dx);
        int idx_y = floor((diff[1] + offset_y) / dy);
        int idx_z = floor((diff[2] + offset_z) / dz);

        return make_tuple(idx_x, idx_y, idx_z);
    }
};

#endif

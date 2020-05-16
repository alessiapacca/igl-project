#ifndef __Uniform_Grid__
#define __Uniform_Grid__

using namespace std;
using namespace Eigen;

class UniformGrid
{
public:
    int resolution = 20;
    Vector3d bb_min;
    Vector3d bb_max;
    Vector3d dim;
    UniformGrid(Vector3d &bb_min, Vector3d& bb_max) {
        this->bb_min = bb_min;
        this->bb_max = bb_max;
        this->dim = bb_max - bb_min;
        this->grid_bins = vector<vector<vector<vector<int>>>>(
            resolution - 1, vector<vector<vector<int>>>(
                resolution - 1, vector<vector<int>>(
                    resolution - 1, vector<int>())));
    }

    UniformGrid() {

    }

    void init_grid(const MatrixXd& P) {       
        // Grid spacing
        for (int i = 0; i < P.rows(); i++) {
            // for each P find enclosing grid cell
            int x = floor((P(i, 0) - bb_min(0)) / dim(0) * (double) (resolution - 2));
            int y = floor((P(i, 1) - bb_min(1)) / dim(1) * (double) (resolution - 2));
            int z = floor((P(i, 2) - bb_min(2)) / dim(2) * (double) (resolution - 2));
            
            grid_bins[x][y][z].push_back(i);
        }
    }
    double query(const VectorXd& P, const MatrixXd& V) {
        int min_idx = -1;
        int max_idx = 1;
        int x0 = floor((P(0) - bb_min(0)) / dim(0) * (double) (resolution - 2));
        int y0 = floor((P(0) - bb_min(1)) / dim(1) * (double) (resolution - 2));
        int z0 = floor((P(0) - bb_min(2)) / dim(2) * (double) (resolution - 2));

        double min_dist = __DBL_MAX__;
        for (int x = min_idx; x <= max_idx; x++) {
            for (int y = min_idx; y <= max_idx; y++) {
                for (int z = min_idx; z <= max_idx; z++) {
                    if (x0 + x >= 0 && x0 + x < resolution - 1
                            && y0 + y >= 0 && y0 + y < resolution - 1
                            && z0 + z >= 0 && z0 + z < resolution - 1) {
                        vector<int> closest_points = grid_bins[x0 + x][y0 + y][z0 + z];
                        if (closest_points.size() == 0) {
                            continue;
                        }
                        for (int i = 0; i < closest_points.size(); i++) {
                            double dist = (P - V.row(closest_points[i])).norm();
                            if (dist < min_dist) {
                                min_dist = dist;
                            }
                        }
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
    vector<vector<vector<vector<int>>>> grid_bins;
};

#endif

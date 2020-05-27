#include <igl/cat.h>
#include <igl/cotmatrix.h>
#include <igl/repdiag.h>
#include <igl/boundary_loop.h>
#include <igl/adjacency_list.h>
#include "uniform_grid.h"

#include <unordered_set>

using namespace std;
using namespace Eigen;

// 3 types of constraints
// (1) landmarks
// (2) boundary points
// (3) points that are close enough to target face
void ConvertConstraintsToMatrixForm(MatrixXd V,
                                    VectorXi indices,
                                    MatrixXd positions,
                                    Eigen::SparseMatrix<double> &C,
                                    VectorXd &d)
{
	C = Eigen::SparseMatrix<double>(indices.rows() * 3, V.rows() * 3);
	C.setZero();
	d = Eigen::VectorXd(indices.rows() * 3);
	d.setZero();

	for (int i = 0; i < indices.rows(); i++) {
		// x
		C.insert(i, indices(i)) = 1;
		d(i) = positions(i, 0);
		// y
		C.insert(indices.rows() + i, V.rows() + indices(i)) = 1;
		d(indices.rows() + i) = positions(i, 1);
		// z
		C.insert(indices.rows() * 2 + i, V.rows() * 2 + indices(i)) = 1;
		d(indices.rows() * 2 + i) = positions(i, 2);
	}
}


int non_rigid_warping(const VectorXi& prior_constraints,
                    const MatrixXd& prior_constraint_positions,
                    unordered_set<int> existing_constraints, 
                    unordered_set<int> clst_constraints, 
                    MatrixXd& V_temp,
                    const MatrixXi& F_temp, 
                    const MatrixXd& V,
                    UniformGrid& ug,
                    const double threshold) {
    
    // ********************************
    // INPUT
    // prior_constraints: indices of boundary vertices and landmarks of the template
    // prior_constraint_positions: coordinates of boundary vertices and landmarks of the template
    // existing_constraints: indices of boundary vertices and landmarks of the template,
    //                       skip these points when adding closest point constraints
    //                       use unordered_set for faster search
    // clst_constraints: indices of the scanned mesh vertices,
    //                   which are already closest points of other template vertices,
    //                   avoid fixing multiple vertices to the same position
    // (V_temp, F_temp): template mesh
    // ug: uniform grid data structure initialized with the target mesh
    // threshold: threshold distance for including a vertex as a constraint
    // 
    // OUTPUT
    // number of vertices that satisfy closest point condition
    // (i.e., distance to its closest point within threshold)
    // ********************************

    VectorXi f;
    SparseMatrix<double> L, A, c;
    VectorXd x_prime, b(V_temp.rows() * 3), d;

    igl::cotmatrix(V_temp, F_temp, L);

    // b = Lx
    b << L * V_temp.col(0), L * V_temp.col(1), L * V_temp.col(2);

    igl::repdiag(L, 3, A);

    VectorXi closest_vertex_indices(V_temp.rows());
    MatrixXd closest_vertex_positions(V_temp.rows(), 3);

    int idx;
    int cnt = 0;
    for (int i = 0; i < V_temp.rows(); i++) {

        if (existing_constraints.find(i) != existing_constraints.end()) continue;

        double dist = ug.query_xyz(V_temp.row(i), V, idx, threshold);

        if (dist < threshold && dist != -1) {
            if (clst_constraints.find(idx) != clst_constraints.end()) continue;
            closest_vertex_indices(cnt) = i;
            closest_vertex_positions.row(cnt) = V.row(idx);
            clst_constraints.insert(idx);
            cnt++;
        }
    }

    closest_vertex_indices.conservativeResize(cnt);
    closest_vertex_positions.conservativeResize(cnt, 3);

    // cout << "#closest point constraint " << closest_vertex_indices.rows() << endl;

    VectorXi all_constraints;
    MatrixXd all_constraint_positions;
    igl::cat(1, closest_vertex_indices, prior_constraints, all_constraints);
    igl::cat(1, closest_vertex_positions, prior_constraint_positions, all_constraint_positions);

    ConvertConstraintsToMatrixForm(V_temp, all_constraints, all_constraint_positions, c, d);

    SparseLU<SparseMatrix<double, ColMajor>, COLAMDOrdering<int> > solver;

    Eigen::SparseMatrix<double, ColMajor> C_T = c.transpose();

    Eigen::SparseMatrix<double> zeros_c(c.rows(), C_T.cols());
    zeros_c.setZero();

    Eigen::SparseMatrix<double, ColMajor> left_side_1, left_side_2, LHS;
    VectorXd RHS; 

    igl::cat(2, A, C_T, left_side_1);
    igl::cat(2, c, zeros_c, left_side_2);
    igl::cat(1, left_side_1, left_side_2, LHS);
    igl::cat(1, b, d, RHS);

    LHS.makeCompressed();
    solver.compute(LHS);

    if (solver.info() != Eigen::Success) {
        cout << "SparseLU Failed!" << endl;
    } 

    x_prime = solver.solve(RHS);

    V_temp.col(0) = x_prime.topRows(V_temp.rows());
    V_temp.col(1) = x_prime.middleRows(V_temp.rows(), V_temp.rows());
    V_temp.col(2) = x_prime.middleRows(2*V_temp.rows(), V_temp.rows());
    
    return closest_vertex_indices.rows();
}
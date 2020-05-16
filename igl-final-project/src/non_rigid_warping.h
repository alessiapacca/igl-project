#include <igl/cat.h>
#include <igl/cotmatrix.h>
#include <igl/repdiag.h>
#include <igl/boundary_loop.h>
#include "uniform_grid.h"

#include <unordered_set>

using namespace std;
using namespace Eigen;

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

void non_rigid_warping(MatrixXd& V_temp,
					   const MatrixXi& F_temp, 
					   const VectorXi& landmarks_temp,
					   const MatrixXd& landmark_positions,
                       const MatrixXd& V,
                       UniformGrid& ug) {
    VectorXi f;
    SparseMatrix<double> L, A, c;
    VectorXd x_prime, b(V_temp.rows() * 3), d;

    igl::cotmatrix(V_temp, F_temp, L);

    // b = Lx
    b << L * V_temp.col(0), L * V_temp.col(1), L * V_temp.col(2);

    // DONE: find a better way to do this? 
    igl::repdiag(L, 3, A);

    // DONE: add boundary points to constraint ************************
    VectorXi boundary_vertex_indices;
    MatrixXd boundary_vertex_positions;
    igl::boundary_loop(F_temp, boundary_vertex_indices);
    igl::slice(V_temp, boundary_vertex_indices, 1, boundary_vertex_positions);

    unordered_set<int> existing_constraints;
    for (int i=0; i<landmarks_temp.rows(); i++) 
        existing_constraints.insert(landmarks_temp(i));
    for (int i=0; i<boundary_vertex_indices.rows(); i++) 
        existing_constraints.insert(boundary_vertex_indices(i));

    VectorXi closest_vertex_indices(V_temp.rows());
    MatrixXd closest_vertex_positions(V_temp.rows(), 3);

    // TODO: find appropiate threshold
    unordered_set<int> clst_constraints;
    double threshold = 2;
    int idx;
    int cnt = 0;
    for (int i = 0; i < V_temp.rows(); i++) {

        if (existing_constraints.find(i) != existing_constraints.end()) continue;

        double dist = ug.query(V_temp.row(i), V, idx, threshold);

        if (dist < threshold && dist != -1) {
            // ADD V_temp.row(i) to constraint;
            if (clst_constraints.find(idx) != clst_constraints.end()) continue;
            closest_vertex_indices(cnt) = i;
            closest_vertex_positions.row(cnt) = V.row(idx);
            clst_constraints.insert(idx);
            cnt++;
        }
    }

    closest_vertex_indices.conservativeResize(cnt);
    closest_vertex_positions.conservativeResize(cnt, 3);

    cout << "closest constraint size " << closest_vertex_indices.rows() << endl;

    VectorXi all_constraints, all_constraints_;
    MatrixXd all_constraint_positions, all_constraint_positions_;
    igl::cat(1, closest_vertex_indices, boundary_vertex_indices, all_constraints_);
    igl::cat(1, all_constraints_, landmarks_temp, all_constraints);
    igl::cat(1, closest_vertex_positions, boundary_vertex_positions, all_constraint_positions_);
    igl::cat(1, all_constraint_positions_, landmark_positions, all_constraint_positions);

    ConvertConstraintsToMatrixForm(V_temp, all_constraints, all_constraint_positions, c, d);
    cout << "set constraints done\n";

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
    } else {
        cout << "SparseLU Succeeded!" << endl;
    }

    x_prime = solver.solve(RHS);

    // DONE: Add x_prime to ?
    V_temp.col(0) = x_prime.topRows(V_temp.rows());
    V_temp.col(1) = x_prime.middleRows(V_temp.rows(), V_temp.rows());
    V_temp.col(2) = x_prime.middleRows(2*V_temp.rows(), V_temp.rows());
    
}
#include <igl/cat.h>
#include <igl/cotmatrix.h>
#include <igl/repdiag.h>
#include <igl/boundary_loop.h>

using namespace std;
using namespace Eigen;

void ConvertConstraintsToMatrixForm(MatrixXd V, VectorXi indices, MatrixXd positions, Eigen::SparseMatrix<double> &C, VectorXd &d)
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

void non_rigid_warping(MatrixXi F, VectorXi landmark_vertices, MatrixXd landmark_vertex_positions, MatrixXd &V) {
    SparseMatrix<double> L, A, A1, A2, A3, c;
    VectorXd x_prime, b(V.rows() * 3), d;

    igl::cotmatrix(V, F, L);

    // b = Lx
    b << L * V.col(0), L * V.col(1), L * V.col(2);
    cout << "B is " << b.rows() << " " << b.cols() << endl;

    igl::repdiag(L, 3, A);
    cout << "A is " << A.rows() << " " << A.cols() << endl;

    VectorXi boundary_vertex_indices;
    MatrixXd boundary_vertex_positions;
    igl::boundary_loop(F, boundary_vertex_indices);
    igl::slice(V, boundary_vertex_indices, 1, boundary_vertex_positions);

    VectorXi all_constraints;
    MatrixXd all_constraint_positions;
    igl::cat(1, boundary_vertex_indices, landmark_vertices, all_constraints);
    igl::cat(1, boundary_vertex_positions, landmark_vertex_positions, all_constraint_positions);

    ConvertConstraintsToMatrixForm(V, all_constraints, all_constraint_positions, c, d);

    cout << "C is " << c.rows() << " " << c.cols() << endl;
    cout << "D is " << d.rows() << " " << d.cols() << endl;


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

    cout << x_prime << endl;
    V.col(0) = x_prime.topRows(V.rows());
    V.col(1) = x_prime.middleRows(V.rows(), V.rows());
    V.col(2) = x_prime.bottomRows(V.rows());
}
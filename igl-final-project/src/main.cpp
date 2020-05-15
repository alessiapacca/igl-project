#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/slice_into.h>
#include <igl/rotate_by_quat.h>
#include <igl/unproject_onto_mesh.h>
#include "Lasso.h"

#include <igl/cat.h>
#include <igl/cotmatrix.h>
#include <igl/repdiag.h>
#include <igl/boundary_loop.h>

//activate this for alternate UI (easier to debug)
//#define UPDATE_ONLY_ON_UP

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

Viewer viewer;
const char* mesh_filename;
//vertex array, #V x3
Eigen::MatrixXd V(0,3), V_cp(0, 3);
//face array, #F x3
Eigen::MatrixXi F(0,3);

//mouse interaction
enum MouseMode { SELECT, NONE };
MouseMode mouse_mode = NONE;
bool doit = false;
int down_mouse_x = -1, down_mouse_y = -1;

int currently_selected_but_not_applied_vertex = -1;

//for selecting vertices
std::unique_ptr<Lasso> lasso;
//list of currently selected vertices
Eigen::VectorXi selected_v(0,1);

//for saving constrained vertices
//vertex-to-handle index, #V x1 (-1 if vertex is free)
Eigen::VectorXi handle_id(0,1);
//list of all vertices belonging to handles, #HV x1
Eigen::VectorXi handle_vertices(0,1);
//updated positions of handle vertices, #HV x3
Eigen::MatrixXd handle_vertex_positions(0,3);
//index of handle being moved
int moving_handle = -1;
//rotation and translation for the handle being moved
Eigen::Vector3f translation(0,0,0);
Eigen::Vector4f rotation(0,0,0,1.);
typedef Eigen::Triplet<double> T;
//per vertex color array, #V x3
Eigen::MatrixXd vertex_colors;

//function declarations (see below for implementation)
bool load_mesh(string filename);
bool solve(Viewer& viewer);
void get_new_handle_locations();
bool compute_closest_vertex();
Eigen::MatrixXd readMatrix(const char *filename);

bool callback_mouse_down(Viewer& viewer, int button, int modifier);
bool callback_mouse_move(Viewer& viewer, int mouse_x, int mouse_y);
bool callback_mouse_up(Viewer& viewer, int button, int modifier);
bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers);
void onNewHandleID();
void applySelection();

//--------------------------------------------------------
// 12/05/2020 by Yingyan
// rigid alignment draft version
// currently use only the given landmark example

// template vertices and faces
Eigen::MatrixXd V_temp(0,3);
Eigen::MatrixXi F_temp(0,3);
// landmark vertex indices and positions
VectorXi landmarks, landmarks_temp;
MatrixXd landmark_positions(0, 3), landmark_positions_temp(0, 3);

// max number of landmarks
const int MAX_NUM_LANDMARK = 30;

// read landmark from file
VectorXi read_landmarks(const char *filename)
{
    VectorXi landmarks;
    landmarks.setZero(MAX_NUM_LANDMARK);

    fstream fin(filename);
    int vertex_id = 0, landmark_num = 0, cnt = 0;
    while(fin >> vertex_id >> landmark_num) {
        landmarks(landmark_num-1) = vertex_id;
        cnt = max(cnt, landmark_num);
    }

    landmarks.conservativeResize(cnt);

    return landmarks;
}

void write_landmarks(const char* filename) {
    ofstream fout(filename);
    for (int i = 0; i < handle_vertices.rows(); i++) {
        fout << handle_vertices[i] << " " << handle_id[handle_vertices[i]] << endl;
    }
    fout.close();
}

// compute average distance to mean landmark
inline double avg_dist(const MatrixXd& mat)
{
    RowVector3d c = mat.colwise().mean();
    VectorXd dist = (mat.rowwise() - c).colwise().norm();
    return dist.mean();
}

void rigid_alignment()
{
    //load scanned mesh
    load_mesh("../data/landmarks_example/person0_.obj");
    landmarks = read_landmarks("../data/landmarks_example/person0__23landmarks");
    landmark_positions(0, 3);
    igl::slice(V, landmarks, 1, landmark_positions);

    // load template
    igl::read_triangle_mesh("../data/landmarks_example/headtemplate.obj",V_temp,F_temp);
    landmarks_temp = read_landmarks("../data/landmarks_example/headtemplate_23landmarks");
    landmark_positions_temp(0, 3);
    igl::slice(V_temp, landmarks_temp, 1, landmark_positions_temp);

    // center template at (0,0,0)
    RowVector3d centroid_temp = V_temp.colwise().mean();
    V_temp = V_temp.rowwise() - centroid_temp;

    // scale template and update landmark positions
    double d_to_mean_landmark = avg_dist(landmark_positions);
    double d_to_mean_landmark_temp = avg_dist(landmark_positions_temp);
    double scaling_factor = d_to_mean_landmark / d_to_mean_landmark_temp;
    V_temp *= scaling_factor;

    // center at landmark mean
    RowVector3d lm_centroid = landmark_positions.colwise().mean();
    RowVector3d lm_centroid_temp = landmark_positions_temp.colwise().mean();
    V = V.rowwise() - lm_centroid;
    V_temp = V_temp.rowwise() - lm_centroid_temp;
    igl::slice(V, landmarks, 1, landmark_positions);
    igl::slice(V_temp, landmarks_temp, 1, landmark_positions_temp);

    // compute rotation matrix via SVD
    Matrix3d R = landmark_positions.transpose() * landmark_positions_temp;
    JacobiSVD<Matrix3d> svd(R, ComputeThinU | ComputeThinV);
    Matrix3d svd_U = svd.matrixU();
    Matrix3d svd_V = svd.matrixV();
    Matrix3d I = Matrix3d::Identity();
    if((svd_U * svd_V.transpose()).determinant() < 0) I(2, 2) = -1; // check for reflection case
    Matrix3d bestRotation = svd_U * I * svd_V.transpose();
    V *= bestRotation;

}
//-----------------------------------------------------------

void ConvertConstraintsToMatrixForm(VectorXi indices, MatrixXd positions, Eigen::SparseMatrix<double> &C, VectorXd &d)
{
	// Convert the list of fixed indices and their fixed positions to a linear system
	// Hint: The matrix C should contain only one non-zero element per row and d should contain the positions in the correct order.
	C = Eigen::SparseMatrix<double>(indices.rows() * 3, V_temp.rows() * 3);
	C.setZero();
	d = Eigen::VectorXd(indices.rows() * 3);
	d.setZero();

	for (int i = 0; i < indices.rows(); i++) {
		// x
		C.insert(i, indices(i)) = 1;
		d(i) = positions(i, 0);
		// y
		C.insert(indices.rows() + i, V_temp.rows() + indices(i)) = 1;
		d(indices.rows() + i) = positions(i, 1);
		// z
		C.insert(indices.rows() * 2 + i, V_temp.rows() * 2 + indices(i)) = 1;
		d(indices.rows() * 2 + i) = positions(i, 2);
	}
}

void non_rigid_warping() {
    VectorXi f;
    SparseMatrix<double> L, A, A1, A2, A3, c;
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

    VectorXi all_constraints;
    MatrixXd all_constraint_positions;
    igl::cat(1, boundary_vertex_indices, landmarks_temp, all_constraints);
    igl::cat(1, boundary_vertex_positions, landmark_positions, all_constraint_positions);

    ConvertConstraintsToMatrixForm(all_constraints, all_constraint_positions, c, d);

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
    V_temp.col(2) = x_prime.bottomRows(V_temp.rows());

}

bool solve(Viewer& viewer)
{
    return true;
};

void get_new_handle_locations()
{
    int count = 0;
    for (long vi = 0; vi < V.rows(); ++vi)
        if (handle_id[vi] >= 0)
        {
            Eigen::RowVector3f goalPosition = V.row(vi).cast<float>();

            if (handle_id[vi] == moving_handle) {

            }
            handle_vertex_positions.row(count++) = goalPosition.cast<double>();
        }
}

bool load_mesh(string filename)
{
    igl::read_triangle_mesh(filename,V,F);
    viewer.data().clear();
    viewer.data().set_mesh(V, F);

    viewer.core().align_camera_center(V);
    V_cp = V;
    handle_id.setConstant(V.rows(), 1, -1);
    // Initialize selector
    lasso = std::unique_ptr<Lasso>(new Lasso(V, F, viewer));

    selected_v.resize(0,1);

    return true;
}

int main(int argc, char *argv[])
{
    if(argc != 2) {
        cout << "Usage assignment5 mesh.off>" << endl;
        load_mesh("../data/scanned_faces_cleaned/alain_normal.obj");
        mesh_filename = "../data/scanned_faces_cleaned/alain_normal.obj";
    }
    else
    {
        load_mesh(argv[1]);
        mesh_filename = argv[1];
    }

    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    menu.callback_draw_viewer_menu = [&]()
    {
        // Draw parent menu content
        menu.draw_viewer_menu();

        // Add new group
        if (ImGui::CollapsingHeader("Landmark Controls", ImGuiTreeNodeFlags_DefaultOpen))
        {
            int mouse_mode_type = static_cast<int>(mouse_mode);

            if (ImGui::Combo("Mouse Mode", &mouse_mode_type, "SELECT\0NONE\0"))
            {
                mouse_mode = static_cast<MouseMode>(mouse_mode_type);
            }

            if (ImGui::Button("Clear Current Selection", ImVec2(-1,0)))
            {
                selected_v.resize(0,1);
                viewer.data().clear_points();
                viewer.data().clear_labels();
            }

            if (ImGui::Button("Apply Selection", ImVec2(-1,0)))
            {
                applySelection();
            }

            if (ImGui::Button("Clear Selected Landmarks", ImVec2(-1,0)))
            {
                handle_id.setConstant(V.rows(),1,-1);
                viewer.data().clear_points();
                viewer.data().clear_labels();
            }

             if (ImGui::Button("Save Landmarks to file", ImVec2(-1,0)))
            {
                char landmark_filename[200];
                strcpy(landmark_filename, mesh_filename);
                strcat(landmark_filename, "_landmarks");
                printf("%s\n",landmark_filename);
                write_landmarks(landmark_filename);
            }

            // -----------------------------------------------------
            // 12/05/2020 by Yingyan
            // test rigid alignment
            if (ImGui::Button("Rigid Alignment", ImVec2(-1,0)))
            {
                rigid_alignment();
                MatrixXd V_total(V.rows() + V_temp.rows(), 3);
                MatrixXi F_total(F.rows() + F_temp.rows(), 3);
                V_total << V, V_temp;
                F_total << F, F_temp + MatrixXi::Constant(F_temp.rows(), 3, V.rows());
                viewer.data().clear();
                viewer.data().set_mesh(V_total, F_total);
            }

            if (ImGui::Button("Display Template", ImVec2(-1,0)))
            {
                viewer.data().clear();
                viewer.data().set_mesh(V_temp, F_temp);
            }

            if (ImGui::Button("Display Scanned Mesh", ImVec2(-1,0)))
            {
                viewer.data().clear();
                viewer.data().set_mesh(V, F);
            }
            // ---------------------------------------------------

            if (ImGui::Button("Non-Rigid Warping", ImVec2(-1,0)))
            {
                non_rigid_warping();
            }

            if (ImGui::Button("Display Non-Rigid Result", ImVec2(-1,0)))
            {
                MatrixXd V_total(V.rows() + V_temp.rows(), 3);
                MatrixXi F_total(F.rows() + F_temp.rows(), 3);
                V_total << V, V_temp;
                F_total << F, F_temp + MatrixXi::Constant(F_temp.rows(), 3, V.rows());
                viewer.data().clear();
                viewer.data().set_mesh(V_total, F_total);
            }
        }
    };

    viewer.callback_key_down = callback_key_down;
    viewer.callback_mouse_down = callback_mouse_down;
    viewer.callback_mouse_move = callback_mouse_move;
    viewer.callback_mouse_up = callback_mouse_up;

    viewer.data().point_size = 10;
    viewer.data().show_labels = true;
    viewer.core().set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
    viewer.launch();
}


bool callback_mouse_down(Viewer& viewer, int button, int modifier)
{
    if (button == (int) Viewer::MouseButton::Right)
        return false;

    down_mouse_x = viewer.current_mouse_x;
    down_mouse_y = viewer.current_mouse_y;

    if (mouse_mode == SELECT)
    {
       if (compute_closest_vertex()) {
            // paint hit red
            VectorXi previous_selected_v = selected_v.block(0, 0, selected_v.rows(), selected_v.cols());
            selected_v.resize(selected_v.rows() + 1, 1);
            selected_v.block(0, 0, previous_selected_v.rows(), previous_selected_v.cols()) = previous_selected_v;
            selected_v(selected_v.rows() - 1) = currently_selected_but_not_applied_vertex;
            MatrixXd selected_v_pos;
            igl::slice(V, selected_v, 1, selected_v_pos);
            viewer.data().set_points(selected_v_pos,Eigen::RowVector3d(1,0,0));
            vector<string> labels = vector<string>(selected_v_pos.rows());
            for (int i = 0; i < labels.size(); i++) {
                labels[i] = std::to_string(i);
            }
            viewer.data().set_labels(selected_v_pos, labels);
       }
    }

    return doit;
}

bool callback_mouse_move(Viewer& viewer, int mouse_x, int mouse_y)
{
    // Currently no mouse move action
    return false;
}

bool callback_mouse_up(Viewer& viewer, int button, int modifier)
{
    // if (!doit)
    //     return false;
    // doit = false;
    // if (mouse_mode == SELECT)
    // {
    //     selected_v.resize(0,1);
    //     lasso->strokeFinish(selected_v);
    //     return true;
    // }

    return false;
};



bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers)
{
    bool handled = false;
    if (key == 'A')
    {
        applySelection();
        callback_key_down(viewer, '1', 0);
        handled = true;
    }
    if (key == 'S')
    {
        mouse_mode = SELECT;
        handled = true;
    }

    //viewer.ngui->refresh();
    return handled;
}

void onNewHandleID()
{
    //store handle vertices too
    int numFree = (handle_id.array() == -1).cast<int>().sum();
    int num_handle_vertices = V.rows() - numFree;
    handle_vertices.setZero(num_handle_vertices);
    handle_vertex_positions.setZero(num_handle_vertices,3);

    int count = 0;
    for (long vi = 0; vi<V.rows(); ++vi)
        if(handle_id[vi] >=0) {
            handle_vertex_positions.row(count) = V.row(vi);
            handle_vertices[count++] = vi;
        }

}

void applySelection()
{
    int index = handle_id.maxCoeff()+1;
    for (int i =0; i<selected_v.rows(); ++i)
    {
        const int selected_vertex = selected_v[i];
        if (handle_id[selected_vertex] == -1) {
            handle_id[selected_vertex] = index;
            index++;
        }
    }
    currently_selected_but_not_applied_vertex = -1;
    onNewHandleID();

    viewer.data().set_points(handle_vertex_positions,Eigen::RowVector3d(0,1,0));
    selected_v.resize(0,1);
    vector<string> labels = vector<string>(handle_vertex_positions.rows());
    for (int i = 0; i < labels.size(); i++) {
        labels[i] = std::to_string(i);
    }
    viewer.data().set_labels(handle_vertex_positions, labels);
}

bool compute_closest_vertex()
{
    int fid;
    Eigen::Vector3f bc;
    double x = down_mouse_x;
    double y = viewer.core().viewport(3) - down_mouse_y;
    if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core().view,
      viewer.core().proj, viewer.core().viewport, V, F, fid, bc))
    {
        currently_selected_but_not_applied_vertex = F(fid, 0);
        return true;
    }
    return false;
}
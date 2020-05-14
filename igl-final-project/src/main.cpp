#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/slice_into.h>
#include <igl/rotate_by_quat.h>
#include <igl/unproject_onto_mesh.h>
#include "Lasso.h"
#include <igl/massmatrix.h>
#include <igl/cotmatrix.h>
#include <igl/slice.h>
#include <stdbool.h>
#include <igl/cat.h>
//activate this for alternate UI (easier to debug)
#define UPDATE_ONLY_ON_UP

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
MouseMode mouse_mode_smooth = NONE;

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
Eigen::MatrixXd handle_centroids(0,3);
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
SparseMatrix<double> Aff, Afc;
Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>, Eigen::RowMajor> solver1;
bool preFactorization = true;
bool show_restVertices = false;
Eigen::MatrixXd d;
Eigen::VectorXi rest_Vertices;
Eigen::MatrixXd B;

//function declarations (see below for implementation)
bool load_mesh(string filename);
bool solve(Viewer& viewer);
void get_new_handle_locations();
bool compute_closest_vertex();
void ConvertConstraintsToMatrixForm(VectorXi indices, MatrixXd positions, Eigen::SparseMatrix<double> &C, VectorXd &d);
Eigen::MatrixXd readMatrix(const char *filename);

bool callback_mouse_down(Viewer& viewer, int button, int modifier);
bool callback_mouse_move(Viewer& viewer, int mouse_x, int mouse_y);
bool callback_mouse_up(Viewer& viewer, int button, int modifier);
bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers);
void onNewHandleID();
void applySelection();

#define MAXNUMREGIONS 7
extern double regionColors[MAXNUMREGIONS][3];
double regionColors[MAXNUMREGIONS][3]= {
        {0, 0.4470, 0.7410},
        {0.8500, 0.3250, 0.0980},
        {0.9290, 0.6940, 0.1250},
        {0.4940, 0.1840, 0.5560},
        {0.4660, 0.6740, 0.1880},
        {0.3010, 0.7450, 0.9330},
        {0.6350, 0.0780, 0.1840},
};

//--------------------------------------------------------
// 12/05/2020 by Yingyan
// rigid alignment draft version
// currently use only the given landmark example

// template vertices and faces
Eigen::MatrixXd V_temp(0,3);
Eigen::MatrixXi F_temp(0,3);

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
    VectorXi landmarks = read_landmarks("../data/landmarks_example/person0__23landmarks");
    MatrixXd landmark_positions(0, 3);
    igl::slice(V, landmarks, 1, landmark_positions);

    // load template
    igl::read_triangle_mesh("../data/landmarks_example/headtemplate.obj",V_temp,F_temp);
    VectorXi landmarks_temp = read_landmarks("../data/landmarks_example/headtemplate_23landmarks");
    MatrixXd landmark_positions_temp(0, 3);
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

void ConvertConstraintsToMatrixForm(VectorXi indices, MatrixXd positions, Eigen::SparseMatrix<double> &C, VectorXd &d)
{
	// Convert the list of fixed indices and their fixed positions to a linear system
	// Hint: The matrix C should contain only one non-zero element per row and d should contain the positions in the correct order.
	C = Eigen::SparseMatrix<double>(indices.rows() * 3, V.rows() * 3);
	C.reserve(3 * indices.size());
	C.setZero();
	d = Eigen::VectorXd(indices.rows() * 3);
	d.setZero();

	for (int i = 0; i < indices.rows(); i++) {
		// x
		C.coeffRef(i, indices(i)) = 1;
		d(i) = positions(i, 0);
		// y
		C.coeffRef(indices.rows() + i, V.rows() + indices(i)) = 1;
		d(indices.rows() + i) = positions(i, 1);
        // x
		C.coeffRef(i, indices(i)) = 1;
		d(i) = positions(i, 0);
		// z
		C.coeffRef(indices.rows() * 2 + i, V.rows() * 2 + indices(i)) = 1;
		d(indices.rows() * 2 + i) = positions(i, 2);
	}
}

void non_rigid_warping() {
    VectorXi f;
    SparseMatrix<double> L, A, A1, A2, A3, c, zeros;
    VectorXd x_prime, b(V.rows() * 3), d;

    igl::cotmatrix(V, F, L);

    // b = Lx
    b << L * V.col(0), L * V.col(1), L * V.col(2);

    // TODO find a better way to do this? 
    zeros = SparseMatrix<double>(L.rows(), L.cols());
    zeros.setZero();
    
    igl::cat(2, L, zeros, A1);
    igl::cat(2, A1, zeros, A1);
    igl::cat(2, zeros, L, A2);
    igl::cat(2, A2, zeros, A2);
    igl::cat(2, zeros, zeros, A3);
    igl::cat(2, zeros, L, A3);
    igl::cat(1, A1, A2, A);
    igl::cat(1, A, A3, A);


    // TODO: add boundary points to constraint ************************
    VectorXi all_constraints = handle_vertices;
    MatrixXd all_constraint_positions = handle_vertex_positions;

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

    // TODO: Add x_prime to ?
}
//-----------------------------------------------------------

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


bool callback_mouse_down(Viewer& viewer, int button, int modifier)
{
    if (button == (int) Viewer::MouseButton::Right)
        return false;

    down_mouse_x = viewer.current_mouse_x;
    down_mouse_y = viewer.current_mouse_y;

    if (mouse_mode_smooth == SELECT)
    {
        if (lasso->strokeAdd(viewer.current_mouse_x, viewer.current_mouse_y) >=0)
            doit = true;
        else
            lasso->strokeReset();
    }
    else if (mouse_mode == SELECT)
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
        }
    }

    return doit;
}



bool callback_mouse_move(Viewer& viewer, int mouse_x, int mouse_y)
{
    if (!doit)
        return false;
    if (mouse_mode_smooth == SELECT)
    {
        lasso->strokeAdd(mouse_x, mouse_y);
        return true;
    }

    return false;
}

bool callback_mouse_up(Viewer& viewer, int button, int modifier)
{
    if (!doit)
        return false;
    doit = false;
    if (mouse_mode_smooth == SELECT)
    {
        selected_v.resize(0,1);
        lasso->strokeFinish(selected_v);
        return true;
    }

    return false;
};


bool callback_pre_draw(Viewer& viewer)
{
    // initialize vertex colors
    vertex_colors = Eigen::MatrixXd::Constant(V.rows(),3,.9);

    // first, color constraints
    int num = handle_id.maxCoeff();
    if (num == 0)
        num = 1;
    for (int i = 0; i<V.rows(); ++i)
        if (handle_id[i]!=-1)
        {
            int r = handle_id[i] % MAXNUMREGIONS;
            vertex_colors.row(i) << regionColors[r][0], regionColors[r][1], regionColors[r][2];
        }
    // then, color selection
    for (int i = 0; i<selected_v.size(); ++i)
        vertex_colors.row(selected_v[i]) << 131./255, 131./255, 131./255.;

    //cout << V.rows() << " and " << vertex_colors.rows() << endl;
    viewer.data().set_colors(vertex_colors);
    viewer.data().V_material_specular.fill(0);
    viewer.data().V_material_specular.col(3).fill(1);
    viewer.data().dirty |= igl::opengl::MeshGL::DIRTY_DIFFUSE | igl::opengl::MeshGL::DIRTY_SPECULAR;



    //clear points and lines
    viewer.data().set_points(Eigen::MatrixXd::Zero(0,3), Eigen::MatrixXd::Zero(0,3));
    viewer.data().set_edges(Eigen::MatrixXd::Zero(0,3), Eigen::MatrixXi::Zero(0,3), Eigen::MatrixXd::Zero(0,3));

    //draw the stroke of the selection
    for (unsigned int i = 0; i<lasso->strokePoints.size(); ++i)
    {
        viewer.data().add_points(lasso->strokePoints[i],Eigen::RowVector3d(0.4,0.4,0.4));
        if(i>1)
            viewer.data().add_edges(lasso->strokePoints[i-1], lasso->strokePoints[i], Eigen::RowVector3d(0.7,0.7,0.7));
    }

    // update the vertex position all the time
    viewer.data().V.resize(V.rows(),3);
    viewer.data().V << V;

    viewer.data().dirty |= igl::opengl::MeshGL::DIRTY_POSITION;

#ifdef UPDATE_ONLY_ON_UP
    //draw only the moving parts with a white line
    if (moving_handle>=0)
    {
        Eigen::MatrixXd edges(3*F.rows(),6);
        int num_edges = 0;
        for (int fi = 0; fi<F.rows(); ++fi)
        {
            int firstPickedVertex = -1;
            for(int vi = 0; vi<3 ; ++vi)
                if (handle_id[F(fi,vi)] == moving_handle)
                {
                    firstPickedVertex = vi;
                    break;
                }
            if(firstPickedVertex==-1)
                continue;


            Eigen::Matrix3d points;
            for(int vi = 0; vi<3; ++vi)
            {
                int vertex_id = F(fi,vi);
                if (handle_id[vertex_id] == moving_handle)
                {
                    int index = -1;
                    // if face is already constrained, find index in the constraints
                    (handle_vertices.array()-vertex_id).cwiseAbs().minCoeff(&index);
                    points.row(vi) = handle_vertex_positions.row(index);
                }
                else
                    points.row(vi) =  V.row(vertex_id);

            }
            edges.row(num_edges++) << points.row(0), points.row(1);
            edges.row(num_edges++) << points.row(1), points.row(2);
            edges.row(num_edges++) << points.row(2), points.row(0);
        }
        edges.conservativeResize(num_edges, Eigen::NoChange);
        viewer.data().add_edges(edges.leftCols(3), edges.rightCols(3), Eigen::RowVector3d(0.9,0.9,0.9));

    }
#endif
    return false;

}




int main(int argc, char *argv[])
{
    if(argc != 2) {
        cout << "Usage igl project mesh.off>" << endl;
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


        if (ImGui::CollapsingHeader("Smoothness Controls", ImGuiTreeNodeFlags_DefaultOpen))
        {
            int mouse_mode_type = static_cast<int>(mouse_mode_smooth);

            if (ImGui::Combo("Mouse Mode", &mouse_mode_type, "SELECT\0NONE\0"))
            {
                mouse_mode_smooth = static_cast<MouseMode>(mouse_mode_type);
            }

            if (ImGui::Button("Clear Selection", ImVec2(-1,0)))
            {
                selected_v.resize(0,1);
            }

            if (ImGui::Button("Apply Selection", ImVec2(-1,0)))
            {
                applySelection();
                preFactorization = true;
            }

            if (ImGui::Button("Clear Constraints", ImVec2(-1,0)))
            {
                handle_id.setConstant(V.rows(),1,-1);
            }

            if(ImGui::Button("Show B", ImVec2(-1,0))){
                show_restVertices = true;
                solve(viewer);
            }
            //ImGui::Checkbox("Show B (rest vertices)", &show_restVertices);
        }
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
        }
    };

    viewer.callback_key_down = callback_key_down;
    viewer.callback_mouse_down = callback_mouse_down;
    viewer.callback_mouse_move = callback_mouse_move;
    viewer.callback_mouse_up = callback_mouse_up;
    viewer.callback_pre_draw = callback_pre_draw;

    viewer.data().point_size = 10;
    viewer.data().show_labels = true;
    viewer.core().set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
    viewer.launch();
}


bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers)
{
    bool handled = false;
    if (key == 'A')
    {
        applySelection();
        preFactorization = true;
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

void compute_handle_centroids()
{
    //compute centroids of handles
    int num_handles = handle_id.maxCoeff()+1;
    handle_centroids.setZero(num_handles,3);

    Eigen::VectorXi num; num.setZero(num_handles,1);
    for (long vi = 0; vi<V.rows(); ++vi)
    {
        int r = handle_id[vi];
        if ( r!= -1)
        {
            handle_centroids.row(r) += V.row(vi);
            num[r]++;
        }
    }

    for (long i = 0; i<num_handles; ++i)
        handle_centroids.row(i) = handle_centroids.row(i).array()/num[i];

}


void onNewHandleID()
{
    //store handle vertices too
    int numFree = (handle_id.array() == -1).cast<int>().sum();
    int num_handle_vertices = V.rows() - numFree;
    handle_vertices.setZero(num_handle_vertices);
    handle_vertex_positions.setZero(num_handle_vertices,3);
    rest_Vertices.resize(V.rows() - handle_vertices.rows());
    int count = 0;
    int count_rest = 0;
    for (long vi = 0; vi<V.rows(); ++vi)
        if(handle_id[vi] >=0) {
            handle_vertex_positions.row(count) = V.row(vi);
            handle_vertices[count++] = vi;
        } else
            rest_Vertices[count_rest++] = vi;

    compute_handle_centroids();
}


//TODO check why index ++
void applySelection()
{
    int index = handle_id.maxCoeff()+1;
    for (int i =0; i<selected_v.rows(); ++i)
    {
        const int selected_vertex = selected_v[i];
        if (handle_id[selected_vertex] == -1)
            handle_id[selected_vertex] = index;
            //index++;
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

void compute_Aff_Afc(SparseMatrix<double> A, SparseMatrix<double> L, SparseMatrix<double> M_inv, MatrixXd & vc, MatrixXd & vf){
    A = Eigen::SparseMatrix<double>(L * M_inv * L);
    igl::slice(A, rest_Vertices, rest_Vertices, Aff);
    igl::slice(A, rest_Vertices, handle_vertices, Afc);
    igl::slice(V, handle_vertices, 1, vc); //vc are the original vertices positions of the ones we have to handle
    solver1.compute(Aff);
    // Aff * vf = -Afc * vf
    MatrixXd b = -Afc * vc;
    vf = solver1.solve(b);
}


void fill_B(MatrixXd vf){
    B.resize(V.rows(),3);
    /* copy V into B */
    B = V.replicate(1, 1);
    igl::slice_into(vf, rest_Vertices, 1, B); //the new smoothed mesh, B[rest vertices] = vf, so we modified the rest vertices
    /* cout << "vf size is: " << vf.rows() << "and " << vf.cols() << endl;
    cout << "vf is: " << vf <<endl;
    cout << "B shape is: " << B.rows() << "and " << B.cols() << endl;
    cout << "B is: " << B <<endl;
    cout << "rest vertices are: " << rest_Vertices << endl; */
}


void show_Vertices(){
    if (show_restVertices)
    {
        V = B;
    }
}

void invert_M(SparseMatrix<double> & M, SparseMatrix<double> & M_inv){
    SimplicialLLT<SparseMatrix<double>> solver;
    solver.compute(M);
    SparseMatrix<double> I(M.rows(),M.cols());
    I.setIdentity();
    M_inv = solver.solve(I);
}



void preFactor() {
    /* initialize data structures */
    preFactorization = false;
    SparseMatrix<double> L, M, A;
    MatrixXd vc;
    MatrixXd vf;
    B.resize(V.rows(), 3);
    Eigen::MatrixXd N;

    /* we need v.transpose() * L * M.inverse() * L * v */
    /* L is the cotangent laplacian of S */
    igl::cotmatrix(V, F, L);
    /* M is the mass matrix */
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_DEFAULT, M);

    /* invert a sparse matrix */
    SparseMatrix<double> M_inv;
    invert_M(M, M_inv);

    /* now let's compute the product of the terms following the slides, we have to find Aff and Afc */
    compute_Aff_Afc(A, L, M_inv, vc, vf);

    /* now let's fill B so that we obtain the new smoothed mesh (free of high frequency details) */
    fill_B(vf);
}



bool solve(Viewer& viewer)
{
    /**** Add your code for computing the deformation from handle_vertex_positions and handle_vertices here (replace following line) ****/
    if(preFactorization)
        preFactor();
    /* now we have
    - rest_Vertices
    - d
    - Aff, Afc initialized */
    show_Vertices();
    return true;
}



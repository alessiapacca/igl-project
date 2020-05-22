#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/slice_into.h>
#include <igl/rotate_by_quat.h>
#include <igl/unproject_onto_mesh.h>
#include "Lasso.h"
#include "non_rigid_warping.h"
#include <igl/cat.h>
#include <igl/cotmatrix.h>
#include <igl/repdiag.h>
#include <igl/boundary_loop.h>
#include "uniform_grid.h"

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

// template vertices and faces
Eigen::MatrixXd V_temp(0,3);
Eigen::MatrixXi F_temp(0,3);
// landmark vertex indices and positions
VectorXi landmarks, landmarks_temp;
MatrixXd landmark_positions(0, 3), landmark_positions_temp(0, 3);
// threshold for closest point distance
double threshold = 1;
// resolution for uniform grid
int xres = 30, yres = 50, zres = 50;

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
    igl::slice(V, landmarks, 1, landmark_positions);

    // load template
    igl::read_triangle_mesh("../data/landmarks_example/headtemplate.obj",V_temp,F_temp);    landmarks_temp = read_landmarks("../data/landmarks_example/headtemplate_23landmarks");
    igl::slice(V_temp, landmarks_temp, 1, landmark_positions_temp);

    // center template at (0,0,0)
    RowVector3d centroid_temp = V_temp.colwise().mean();
    V_temp = V_temp.rowwise() - centroid_temp;

    // scale template and update landmark positions
    double d_to_mean_landmark = avg_dist(landmark_positions);
    double d_to_mean_landmark_temp = avg_dist(landmark_positions_temp);
    double scaling_factor = d_to_mean_landmark / d_to_mean_landmark_temp;
    V_temp *= scaling_factor;
    igl::slice(V_temp, landmarks_temp, 1, landmark_positions_temp);

    // center at landmark mean
    RowVector3d lm_centroid = landmark_positions.colwise().mean();
    RowVector3d lm_centroid_temp = landmark_positions_temp.colwise().mean();
    V = V.rowwise() - lm_centroid;
    V_temp = V_temp.rowwise() - lm_centroid_temp;
    igl::slice(V, landmarks, 1, landmark_positions);
    igl::slice(V_temp, landmarks_temp, 1, landmark_positions_temp);

    // compute rotation matrix via SVD
    MatrixXd R = landmark_positions.transpose() * landmark_positions_temp;
    JacobiSVD<MatrixXd> svd(R, ComputeThinU | ComputeThinV);
    Matrix3d svd_U = svd.matrixU();
    Matrix3d svd_V = svd.matrixV();
    Matrix3d I = Matrix3d::Identity();
    if((svd_U * svd_V.transpose()).determinant() < 0) I(2, 2) = -1; // check for reflection case
    Matrix3d bestRotation = svd_U * I * svd_V.transpose();
    V *= bestRotation;

    // update landmark positions
    igl::slice(V, landmarks, 1, landmark_positions);

}

int nb_eigenfaces = 8;
Eigen::MatrixXd mean_face_V;
Eigen::MatrixXi mean_face_F;
std::vector<Eigen::MatrixXd> eigen_faces;

VectorXd weights_morph_f1, weights_morph_f2;
VectorXd mean_face_flatten;
std::vector<VectorXd> eigen_faces_flatten;
std::vector<VectorXd> original_faces_flatten;

VectorXd eigen_values;
std::vector<float> eigen_face_weights(nb_eigenfaces, 0.0);
bool has_svd_run = false;
bool has_initialized_morph = false;

// The eigen_faces pca. Need the faces to be used. Each face is represented by its set of vertices.
void pca_eigenfaces(std::vector<Eigen::MatrixXd> faces){

    std::cout << "PCA start!" << std::endl;

    original_faces_flatten.clear();
    eigen_faces_flatten.clear();
    eigen_faces.clear();

    int S = faces.size();
    if (nb_eigenfaces > S){
        std::cout << "Problem: please ask for less eigen vectors\n";
        return;
    }

    // Computing the mean face
    mean_face_V = Eigen::MatrixXd::Zero(faces[0].rows(), 3);
    for (int i=0; i<S; i++){
        mean_face_V = mean_face_V + faces[i];
    }
    mean_face_V = mean_face_V/S;

    std::cout << "Mean Face Calculated!" << std::endl;

    // Computing the covariance matrix
    MatrixXd C = Eigen::MatrixXd::Zero(faces[0].rows()*3, faces[0].rows()*3); // 3#V * 3#V matrix

    mean_face_flatten.setZero(3*mean_face_V.rows());
    for (int j = 0; j< mean_face_V.rows(); j++){
        mean_face_flatten[3*j+0] = mean_face_V.row(j)[0];
        mean_face_flatten[3*j+1] = mean_face_V.row(j)[1];
        mean_face_flatten[3*j+2] = mean_face_V.row(j)[2];
    }

    for (int i=0; i<S; i++){
        VectorXd face_flatten(3*faces[i].rows());
        for (int j = 0; j< faces[i].rows(); j++){
            face_flatten[3*j+0] = faces[i].row(j)[0];
            face_flatten[3*j+1] = faces[i].row(j)[1];
            face_flatten[3*j+2] = faces[i].row(j)[2];
        }
        original_faces_flatten.push_back(face_flatten);
        VectorXd centered_face = face_flatten - mean_face_flatten;
        C += centered_face * centered_face.transpose();
    }

    C/=S;

    std::cout << "Covariance calculated" << std::endl;

    JacobiSVD<MatrixXd> svd(C, ComputeThinU | ComputeThinV);
    auto sv = svd.singularValues();
    eigen_values = VectorXd(nb_eigenfaces);
    for (int i=0; i<nb_eigenfaces; i++){
        eigen_values.row(i) = sv.row(i);
    }
    std::cout << "Eigen Values calculated" << std::endl;
    MatrixXd eigen_vectors = svd.matrixU();
    for (int i=0; i<nb_eigenfaces; i++){
        Eigen::MatrixXd eigen_vector_current = eigen_vectors.col(i);
        MatrixXd eigen_face(faces[0].rows(), 3);
        for (int j = 0; j< faces[0].rows(); j++){
            eigen_face.row(j) << eigen_vector_current.row(3*j+0), eigen_vector_current.row(3*j+1), eigen_vector_current.row(3*j+2) ;
        }
        eigen_faces.push_back(eigen_face);
        eigen_faces_flatten.push_back(eigen_vectors.col(i));
        std::cout << "Eigen Face calculated" << std::endl;
    }
    std::cout << "PCA Done" << std::endl;
    has_svd_run = true;
}

void eigen_face_computations(std::vector<std::string> files){
    std::vector<Eigen::MatrixXd> faces;

    Eigen::MatrixXd VV(0,3);
    Eigen::MatrixXi FF(0,3);
    for (auto it = files.begin(); it!=files.end(); it++){
        igl::read_triangle_mesh(*it,VV,FF);
        faces.push_back(VV);
        std::cout << "1 file read!" << std::endl;
    }

    pca_eigenfaces(faces);
    mean_face_F = FF;

    viewer.data().clear();
    viewer.data().set_mesh(mean_face_V, mean_face_F);
}

void eigen_face_update(){
    Eigen::MatrixXd new_face = mean_face_V;
    // std::cout << eigen_values << std::endl;
    for (int i=0; i<nb_eigenfaces; i++){
        new_face += eigen_faces[i] * eigen_face_weights[i] * eigen_values[i] * 0.01;
    }
    viewer.data().clear();
    viewer.data().set_mesh(new_face, mean_face_F);
}

//Not tested yet.
void initialize_morphing(int face_index1, int face_index2)
{
    if (!has_svd_run) {
        std::cout << "Morphing Initialize: First run SVD to calculate the Eigen faces!" << std::endl;
        return;
    }

    int num_faces = original_faces_flatten.size();
    if (face_index1 >= num_faces || face_index2 >= num_faces) {
        std::cout << "Morphing Initialize: Face indices out of bounds!" << std::endl;
        return;
    }

    VectorXd x1 = original_faces_flatten[face_index1] - mean_face_flatten;
    VectorXd x2 = original_faces_flatten[face_index2] - mean_face_flatten;
    VectorXd x1_t = x1.transpose();
    VectorXd x2_t = x2.transpose();

    weights_morph_f1.setZero(nb_eigenfaces);
    weights_morph_f2.setZero(nb_eigenfaces);
    for (int i = 0; i < nb_eigenfaces; i++) {
        weights_morph_f1[i] = x1_t.dot(eigen_faces_flatten[i]);
        weights_morph_f2[i] = x2_t.dot(eigen_faces_flatten[i]);
    }
}

void face_morphing(float weight) {
    //f_morph = mean_face + sum (weigts_morph_f1_i - nb_eigenfaces * (weights_morph_f1_i - weights_morph_f2_i)) * eigen_faces[i]
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

            if (ImGui::Button("Non-Rigid Warping", ImVec2(-1,0)))
            {
                // aligned results (V_total, F_total)
                MatrixXd V_total(V.rows() + V_temp.rows(), 3);
                MatrixXi F_total(F.rows() + F_temp.rows(), 3);
                V_total << V, V_temp;
                F_total << F, F_temp + MatrixXi::Constant(F_temp.rows(), 3, V.rows());

                // prepare uniform grid
                RowVector3d bb_min = V_total.colwise().minCoeff();
                RowVector3d bb_max = V_total.colwise().maxCoeff();
                UniformGrid ug(bb_min, bb_max, xres, yres, zres); // can integrate resolution into UI
                ug.init_grid(V);

                non_rigid_warping(V_temp, F_temp, landmarks, landmarks_temp, landmark_positions, V, ug, threshold);

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

            ImGui::InputDouble("threshold", &threshold, 0, 0);

            ImGui::InputInt("Resolution x", &xres, 0, 0);
            ImGui::InputInt("Resolution y", &yres, 0, 0);
            ImGui::InputInt("Resolution z", &zres, 0, 0);
        }
    };

    // Draw additional windows
    menu.callback_draw_custom_window = [&]()
    {
    // Define next window position + size
    ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiSetCond_FirstUseEver);
    ImGui::Begin(
      "Eigenfaces", nullptr,
      ImGuiWindowFlags_NoSavedSettings
    );

    // Select the folder with all the faces
    if (ImGui::CollapsingHeader("Meshes", ImGuiTreeNodeFlags_DefaultOpen))
    {
        float w = ImGui::GetContentRegionAvailWidth();
        float p = ImGui::GetStyle().FramePadding.x;
        if (ImGui::Button("Run SVD##Meshes", ImVec2((w-p), 0)))
        {
            std::vector<std::string> files {"../data/aligned_faces_example/example1/fabian-brille.objaligned.obj", 
            "../data/aligned_faces_example/example1/fabian-neutral.objaligned.obj", 
            "../data/aligned_faces_example/example1/fabian-smile.objaligned.obj",
            "../data/aligned_faces_example/example1/jan-smile.objaligned.obj",      
            "../data/aligned_faces_example/example1/jan-neutral.objaligned.obj",
            "../data/aligned_faces_example/example1/jan-brille.objaligned.obj",
            "../data/aligned_faces_example/example1/michi-smile.objaligned.obj",
            "../data/aligned_faces_example/example1/michi-brille.objaligned.obj",
            "../data/aligned_faces_example/example1/michi-neutral.objaligned.obj",
            // "../data/aligned_faces_example/example1/selina-smile.objaligned.obj",    
            // "../data/aligned_faces_example/example1/selina-neutral.objaligned.obj",  
            // "../data/aligned_faces_example/example1/selina-brille.objaligned.obj",   
            // "../data/aligned_faces_example/example1/simon-brille.objaligned.obj",    
            // "../data/aligned_faces_example/example1/simon-neutral.objaligned.obj",   
            // "../data/aligned_faces_example/example1/simon-smile.objaligned.obj",       
            // "../data/aligned_faces_example/example1/zsombor-smile.objaligned.obj",
            // "../data/aligned_faces_example/example1/zsombor-brille.objaligned.obj",
            // "../data/aligned_faces_example/example1/zsombor-neutral.objaligned.obj",
            // "../data/aligned_faces_example/example1/livio-brille.objaligned.obj",   
            // "../data/aligned_faces_example/example1/livio-smile.objaligned.obj",    
            // "../data/aligned_faces_example/example1/livio-neutral.objaligned.obj",  
            // "../data/aligned_faces_example/example1/virginia-brille.objaligned.obj",   
            // "../data/aligned_faces_example/example1/virginia-smile.objaligned.obj",
            // "../data/aligned_faces_example/example1/virginia-neutral.objaligned.obj",
            // "../data/aligned_faces_example/example1/nici-brille.objaligned.obj",    
            // "../data/aligned_faces_example/example1/nici-neutral.objaligned.obj",   
            // "../data/aligned_faces_example/example1/nici-smile.objaligned.obj",     

             };
            eigen_face_computations(files);
        }
    }

    if (ImGui::CollapsingHeader("Eigen Faces", ImGuiTreeNodeFlags_DefaultOpen))
    {

        if (ImGui::SliderFloat("Eigenface 1", &eigen_face_weights[0], 0, 100)
        ||  ImGui::SliderFloat("Eigenface 2", &eigen_face_weights[1], 0, 100)
        ||  ImGui::SliderFloat("Eigenface 3", &eigen_face_weights[2], 0, 100)
        ||  ImGui::SliderFloat("Eigenface 4", &eigen_face_weights[3], 0, 100)
        ||  ImGui::SliderFloat("Eigenface 5", &eigen_face_weights[4], 0, 100)
        ||  ImGui::SliderFloat("Eigenface 6", &eigen_face_weights[5], 0, 100)
        ||  ImGui::SliderFloat("Eigenface 7", &eigen_face_weights[6], 0, 100)
        ||  ImGui::SliderFloat("Eigenface 8", &eigen_face_weights[7], 0, 100)){
            eigen_face_update();
        }

    }



    ImGui::End();
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
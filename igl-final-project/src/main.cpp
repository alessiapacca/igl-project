#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/slice_into.h>
#include <igl/rotate_by_quat.h>
#include <igl/unproject_onto_mesh.h>
#include "Lasso.h"

//activate this for alternate UI (easier to debug)
//#define UPDATE_ONLY_ON_UP

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

Viewer viewer;

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
    }
    else
    {
        load_mesh(argv[1]);
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

            if (ImGui::Button("Clear All Selection", ImVec2(-1,0)))
            {
                selected_v.resize(0,1);
                viewer.data().clear_points();
            }

            if (ImGui::Button("Apply Selection", ImVec2(-1,0)))
            {
                applySelection();
            }

            if (ImGui::Button("Clear Current Selection", ImVec2(-1,0)))
            {
                handle_id.setConstant(V.rows(),1,-1);
                viewer.data().clear_points();
            }
        }
    };

    viewer.callback_key_down = callback_key_down;
    viewer.callback_mouse_down = callback_mouse_down;
    viewer.callback_mouse_move = callback_mouse_move;
    viewer.callback_mouse_up = callback_mouse_up;

    viewer.data().point_size = 10;
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
        if (handle_id[selected_vertex] == -1)
            handle_id[selected_vertex] = index;
            index++;
    }
    currently_selected_but_not_applied_vertex = -1;
    onNewHandleID();

    viewer.data().set_points(handle_vertex_positions,Eigen::RowVector3d(0,1,0));
    selected_v.resize(0,1);

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
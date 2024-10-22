import os 
from stl import mesh 
import open3d as o3d
import numpy as np
import math
import pyvista as pv
import subprocess
import gias3.applications.giasrigidreg as giasrigidreg
import argparse
import ownmodules.giassurfacedistance as gsurf
import ownmodules.giastrainpcashapemodel as gtrainpca
import ownmodules.giaspcreg as pcreg
import pyvista as pv
from scipy.spatial import KDTree
import vtk 
import ownmodules.yatpkg.util.data as data
import ownmodules.yatpkg.math.transformation as trans_sphere
import ownmodules.transformation as trans
import pymeshlab
from gias3.mesh import vtktools
import pandas as pd
from helper_ssm import angle_between_vectors, angle_between_vectors_cross, align_meshes_by_rotation, plotter_vector, calculate_neck_shaft_angle , shortest_line_segment, sphere_fit, get_centroid, get_point_clound, angle_between_vectors_2d, ellipsoid_fit
#if left femur need to be flipp turn to 0
create_txt = 0
flip_n_move = 0
put_in_new_coordinate_system = 0
align_pcaxes = 0
cut_to_neck = 0 
cut_to_prox = 0
cut_out_cam = 0
close_holes = 0
#if radial basis/rigid reg/shapemodel wanted turn on
rbreg = 0
rigidreg = 0
trainpcashapemodel = 0
grooming = 0# turn on if segmentations are not groomed (flipped) yet
init_alignment = 0
fit_shapemodel = 0
surface_to_surface = 0
nearest_neighbor_distance = 0
compare_pcs = 0
cut_out_acetabulum = 0
get_point_id_list = 0
get_point_id_list_vtk = 0
define_CAM_state = 0
get_coupled_mesh = 0
get_femur_coordinate_system = 0
get_femur_measurements = 1
center_femur_to_FHC = 0
combine_two_meshes = 0
# change to direction where femur bones are located 

# list of all files located in the 

files = os.listdir()

#dir to shape model path
ssm_path = "F:/Public/12_Femur_VIFM"




os.chdir(ssm_path)

#df = pd.read_csv("04_Fitting_results/csv.csv")

#fitted_meshes = df["File"].to_list()
if create_txt == 1:
    # create .txt file for fitted meshes

    fitted_meshes = os.listdir(os.path.join(ssm_path, '00_Data/05_Proximal_Femur_no_Cam'))
    #fitted_meshes = [femur for femur in fitted_meshes_unfiltered if "CON" in femur]
    # Save the list of text_file strings to a text file
    text_files_file_path = os.path.join(ssm_path, '01_txt_files/No_cam_Data.txt')
    with open(text_files_file_path, 'w') as text_files_file:
        for text_file in fitted_meshes:
            text_file = os.path.join('00_Data/05_Proximal_Femur_no_Cam', (text_file))
            text_files_file.write(text_file + '\n')

# loop through list of files 
if flip_n_move == 1:
    
    files = os.listdir("00_Data/00_Segmentations")
    for f in files: 
        if 'L' in f: 
            #Load mesh data 
            output_file = os.path.splitext(f)[0] + "_flipped.stl"
            
            output_file = os.path.join(ssm_path, '00_Data/01_Flipped/', output_file)

            input_file = os.path.join(ssm_path, "00_Data/00_Segmentations/", f)

            mesh_data = mesh.Mesh.from_file(input_file)

            # Create a transformation matrix for flipping around the X-axis
            flip_x_matrix = np.array([[-1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])

            # Apply the transformation to vertices
            mesh_data.vectors = np.dot(mesh_data.vectors, flip_x_matrix)
            # Invert the face orientation by reversing vertex order
            for i in range(len(mesh_data.vectors)):
                mesh_data.vectors[i] = mesh_data.vectors[i][::-1]
            

        if 'R' in f: 
            #get new direction of file
            output_file = os.path.join(ssm_path, '00_Data/01_Flipped/', f)
            input_file = os.path.join(ssm_path, "00_Data/00_Segmentations/", f)
            mesh_data = mesh.Mesh.from_file(input_file)
            
        # use teds/jus library to get principle axes
        pca_model = trans.PCAModel() 

        try: 

            #mesh_data.transform(trans_mat)
            
            points = mesh_data.points
            centroid = np.mean(np.array(points.reshape(-1,3)), axis=0)

            trans_mat = np.identity(4)
            trans_mat[:3,3] = centroid * (-1)
            mesh_data.transform(trans_mat)

            mesh_data.save(output_file)
            #print(f"STL file saved as {output_file}")
        except:
            print(f"STL file {f} did not work or was not used!")


    # get list of all files in segmentations
    segmentations = os.listdir(os.path.join( ssm_path, '00_Data/01_Flipped'))
    text_file_path = os.path.join( ssm_path, '01_txt_files/raw_list_VIFM.txt')
    # write txt file with all segmentations
    with open(text_file_path, 'w') as text_file:
        for segmentation in segmentations:
            seg_file_name = os.path.join('00_Data/01_Flipped/', segmentation)
            text_file.write(seg_file_name + '\n')

if put_in_new_coordinate_system == 1:
    files = os.listdir("00_Data/06_Proximal_Femur")

    for file in files: 
        f_file = os.path.join(ssm_path, "00_Data/06_Proximal_Femur", file)

        mesh_data = o3d.io.read_point_cloud(f_file)
        femur_mesh = data.VTKMeshUtl.load(f_file)
        points = np.asarray(mesh_data.points) 

        # read shaft data 
        prox_mesh =  os.path.join(ssm_path, "00_Data/03_Shaft", (file[:21] + "_dia_prox.ply"))
        dist_mesh = os.path.join(ssm_path, "00_Data/03_Shaft", (file[:21] + "_dia_dis.ply"))

        mesh_data_prox = o3d.io.read_point_cloud(prox_mesh)
        points_prox = np.asarray(mesh_data_prox.points)

        mesh_data_dist = o3d.io.read_point_cloud(dist_mesh)
        points_dist = np.asarray(mesh_data_dist.points)

        vector_dia =  np.mean(points_prox, axis=0) - np.mean(points_dist, axis= 0)

        # calculate midpoint of neck 
        point_ids_femur_neck = np.loadtxt("01_txt_files/femur_neck.txt")
        # get all coordinates of shaft using the template point id and calculate centroid 
        #get number of Points to iterate over it later
        centroid_neck= get_centroid(femur_mesh, point_ids_femur_neck) 
        # get center of ball
        point_ids_femur_head = np.loadtxt("01_txt_files/femur_head.txt")

        p_cloud_femur_head = get_point_clound(femur_mesh, point_ids_femur_head)

        radius, center_femur_head = sphere_fit(p_cloud_femur_head)

        # use head center and neck center for x axis and vector from diaphysis to build temporary y axis 
        x_axis = (center_femur_head - centroid_neck)
        x_axis = x_axis / np.linalg.norm(x_axis)
        temp_y_axis = vector_dia / np.linalg.norm(vector_dia)
        z_axis = np.cross(x_axis, temp_y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)


        mesh_data = o3d.io.read_triangle_mesh(f_file)

        transformation_matrix = np.identity(4)


        transformation_matrix[:3,3] = -centroid_neck
        mesh_data.transform(transformation_matrix) 

        transformation_matrix = np.identity(4)

        transformation_matrix[0,:3] = x_axis
        transformation_matrix[1,:3] = y_axis
        transformation_matrix[2,:3] = z_axis

        rot_matrix = transformation_matrix[:3,:3]

        print(np.linalg.det(rot_matrix))

        trans_femur_center = center_femur_head @ transformation_matrix[:3,:3]
        

        

        #mesh_data2 = o3d.io.read_triangle_mesh(f_file2)

        mesh_data.transform(transformation_matrix)  
        
        mesh_data.compute_vertex_normals()

        ## Manually flip the normals by negating them
        normals = np.asarray(mesh_data.vertex_normals)
        #mesh_data.vertex_normals = o3d.utility.Vector3dVector(-normals)

        o3d.io.write_triangle_mesh(os.path.join(ssm_path, "00_Data/08_Proximal_Femur_Alpha_align", file), mesh_data)  
        print(transformation_matrix)
        print("Hello, print")  


if align_pcaxes == 1: 
    i = 0
    files = os.listdir(r"segmentations_pcreg")
    for f in files: 
        #ms = pymeshlab.MeshSet()
        #ms.load_new_mesh(os.path.join(ssm_path, "femur_proximal", f))

        #f_file2 = os.path.join(ssm_path, "segmentations_pcreg" ,f)
        f_file = os.path.join(ssm_path, "segmentations_pcreg" ,f)
        if ".stl" in f:
            mesh_data = mesh.Mesh.from_file(f_file)
            points = mesh_data.points
            points = np.array(points.reshape(-1,3))
        else:       
            mesh_data = o3d.io.read_point_cloud(f_file)
            points = np.asarray(mesh_data.points)
        #points = points.reshape(-1, 3)
        # use teds/jus library to get principle axes
        pca_model = trans.PCAModel() 
        output_pca_rot = pca_model.pca_rotation(points)
        rotation_matrix = output_pca_rot.transformation
 

        # get identical orientation after pca trans
        if i == 0:
            
            first_mesh = points @ rotation_matrix.T 
        # get helper function to get same orientation
        else:
            rotation_matrix = align_meshes_by_rotation(first_mesh, points)
        transformation_matrix = np.eye(4)
                
        transformation_matrix[:3,:3] = rotation_matrix
        #print(transformation_matrix)
        mesh_data = o3d.io.read_triangle_mesh(f_file)

        #mesh_data2 = o3d.io.read_triangle_mesh(f_file2)

        mesh_data.transform(transformation_matrix)
        #mesh_data2.transform(transformation_matrix)

        points = np.asarray(mesh_data.vertices)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3,3] =  np.mean(np.array(points.reshape(-1,3)), axis=0) * (-1)

        mesh_data.transform(transformation_matrix)
        #mesh_data2.transform(transformation_matrix)

        mesh_data.compute_vertex_normals()
        #mesh_data2.compute_vertex_normals()

        #o3d.io.write_triangle_mesh(os.path.join(ssm_path, "00_Data/12_Proximal_Femur_Correspondance_align_PCA_align", f), mesh_data2)
        o3d.io.write_triangle_mesh(os.path.join(ssm_path, "griffith_pc_align", f), mesh_data)
        i = i +1
if cut_to_neck == 1: 
    os.chdir(r"D:\Public\05Griffith_Dataset\ShapeModel_allData\fitted_meshes")
    femur_list = os.listdir()

    for femur in femur_list:
        VTKMeshUtl = data.VTKMeshUtl.load(femur)
        #extract point on top of femur head
        Top = np.array(VTKMeshUtl.GetPoint(4061))
        #extract two points on femur neck to shaft connection
        Neck_one = np.array(VTKMeshUtl.GetPoint(2753))
        Neck_two = np.array(VTKMeshUtl.GetPoint(2419))
        mid_neck = (Neck_one+Neck_two)/2

        # creat vectors for coordinate system
        X_axis = (Top - mid_neck) 
        Y_axis_temp = mid_neck - Neck_one
        Z_axis = np.cross(X_axis,Y_axis_temp)
        Y_axis = np.cross(Z_axis, X_axis)
        # get unit vectors
        X_axis /= np.linalg.norm(X_axis)
        Y_axis /= np.linalg.norm(Y_axis)
        Z_axis /= np.linalg.norm(Z_axis)
        # create identity matirx
        tr_matrix = np.identity(4)

        tr_matrix[0,:3] = X_axis
        tr_matrix[1,:3] = Y_axis
        tr_matrix[2,:3] = Z_axis
        #tr_matrix[:3,3] = mid_neck
        output_file = ("D:/Public/05Griffith_Dataset/ShapeModel_allData/cut_to_neck/" + femur)
        # now apply transformation matrix
        mesh_femur = o3d.io.read_triangle_mesh(femur) 
        mesh_femur.transform(tr_matrix)
        mesh_femur.compute_vertex_normals()
        o3d.io.write_triangle_mesh(output_file, mesh_femur)

        #now read in the mesh again 
        VTKMeshUtl = data.VTKMeshUtl.load(output_file)

        #get number of Points to iterate over it later
        num_points = VTKMeshUtl.GetNumberOfPoints()
        p_X = VTKMeshUtl.GetPoint(2753)
        # Create List in which PointID is saved
        PointID_List = []
        #loop over all points in Mesh and check if the z_coordinate is smaller than reference --> if yes save PointID 
        for i in range(num_points): 
            currentPoint = VTKMeshUtl.GetPoint(i)
            if currentPoint[0] > p_X[0]:
                PointID_List.append(i)
        
        Cut_Mesh = data.VTKMeshUtl.sub_mesh(PointID_List, VTKMeshUtl)
        femur_new = (os.path.splitext(femur)[0] + '_neck.ply')
        femur_new = ('D:/Public/05Griffith_Dataset/ShapeModel_allData/cut_to_neck/' + femur_new)
        # write cut mesh in new_Folder
        data.VTKMeshUtl.write(output_file, Cut_Mesh)

    pass
# if you want to use it, make sure to specify reference point and that rbreg was peformed before
if cut_to_prox == 1:
    print_text = 1 
    #os.chdir(r"F:\Public\14_Femur_Griffith\00_Data")

    femur_list = os.listdir()
    femur_list_2 = os.listdir(r"00_Data\02_Radial_Basis_Fit")

    # Create a set of the femur names from femur_list_2, with "_prox" removed
    femur_list_2_basenames = {femur.replace('_2mm_flipped_rbfreg', '') for femur in femur_list_2}
    

    femur_list = {femur.replace('_2mm_rbfreg', '') for femur in femur_list_2_basenames}
    femur_list = list(femur_list)
    #
    # femur_list = [femur for femur in femur_list if femur not in femur_list_2_basenames]
    os.chdir(r"F:\Public\12_Femur_VIFM\00_Data\02_Radial_Basis_Fit")
    #os.chdir("D:/Public/AUT/SSM/aligned_meshes")
    for i, femur in enumerate(femur_list_2):

        VTKMeshUtl = data.VTKMeshUtl.load(femur)

        #get number of Points to iterate over it later
        num_points = VTKMeshUtl.GetNumberOfPoints()
        p_Z = VTKMeshUtl.GetPoint(11556)
        # Create List in which PointID is saved
        PointID_List = []
        #loop over all points in Mesh and check if the z_coordinate is smaller than reference --> if yes save PointID 
        PointID_List = [i for i in range(num_points) if VTKMeshUtl.GetPoint(i)[2] > p_Z[2]]

        # for i in range(num_points): 
        #     currentPoint = VTKMeshUtl.GetPoint(i)
        #     if currentPoint[2] > p_Z:
        #         PointID_List.append(i)
        
        Cut_Mesh = data.VTKMeshUtl.sub_mesh(PointID_List, VTKMeshUtl)
        femur_new = (os.path.splitext(femur_list[i])[0] + '_prox.ply')
        femur_new = ('F:/Public/12_Femur_VIFM/00_Data/06_Proximal_Femur/' + femur_new)
        # write cut mesh in new_Folder
        data.VTKMeshUtl.write(femur_new, Cut_Mesh)
        #open mesh again to close the opening
        #ms = pymeshlab.MeshSet()
        #ms.load_new_mesh(femur_new)
        #ms.apply_coord_laplacian_smoothing(stepsmoothnum=5)
        #ms.meshing_close_holes(maxholesize=100)
        #ms.save_current_mesh(femur_new)
        print("Cut to prox finished")
        # get list of all files in segmentations
   
    if print_text == 1: 
        segmentations = os.listdir('F:\Public/12_Femur_VIFM/00_Data/06_Proximal_Femur')
        text_file_path = os.path.join( ssm_path, '01_txt_files/VIFM_proximal_Femur.txt')
        # write txt file with all segmentations
        with open(text_file_path, 'w') as text_file:
            for segmentation in segmentations:
                seg_file_name = os.path.join('00_Data/06_Proximal_Femur', segmentation)
                text_file.write(seg_file_name + '\n')

if close_holes == 1:
    file_path = "D:/Public/07_CamSSM/pelvis/pelvis_segmenations.txt"
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Strip newline characters and create a list
    segmentations_list = [line.strip() for line in lines]

    for file in segmentations_list: 
        current_file = file[58:]
        print(current_file)

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file)
        try:
            ms.meshing_close_holes(maxholesize=100)
        except:
            print(file, "Cant be closed")
        
        #ms.save_current_mesh(os.path.join("segmentations/",file))
        #ms.load_new_mesh(os.path.join("segmentations/",file))
        ms.apply_coord_laplacian_smoothing(stepsmoothnum=3)
        #ms.meshing_close_holes(maxholesize=30)
        #ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.AbsoluteValue(2))
        ms.save_current_mesh(os.path.join("segmentations/",current_file))
    
    # get list of all files in segmentations
    segmentations = os.listdir(os.path.join( ssm_path, 'segmentations'))
    text_file_path = os.path.join( ssm_path, 'segmentations_groomed.txt')
    # write txt file with all segmentations
    with open(text_file_path, 'w') as text_file:
        for segmentation in segmentations:
            seg_file_name = os.path.join('segmentations/', segmentation)
            text_file.write(seg_file_name + '\n')
    

# creates correspondence on segmentations --> currently doesnt work
if rbreg == 1:
    os.chdir(ssm_path)
    # Create Command line argument
    command = 'gias-rbfreg -b 01_txt_files/raw_data.txt -d 00_Data/02_Radial_Basis_Fit --outext .ply'

    # Run the command
    subprocess.run(command, shell=True)

    # create .txt file for fitted meshes 
    fitted_meshes = os.listdir(os.path.join( ssm_path, '00_Data/02_Radial_Basis_Fit'))

    # Save the list of text_file strings to a text file
    text_files_file_path = os.path.join(ssm_path, '01_txt_files/VIFM_Femur_rbf.txt')
    with open(text_files_file_path, 'w') as text_files_file:
        for text_file in fitted_meshes:
            text_file = os.path.join('00_Data/02_Radial_Basis_Fit', text_file)
            text_files_file.write(text_file + '\n')

# alignes segmentations 
if rigidreg == 1: 
    os.chdir(ssm_path)
    #input class for rigidreg
    class input(): 
        def __init__(self, reg_method, source=None, target=None, out=None, points_only=False, t0=None, r0=None, s0=None, batch=None, outdir=None, outext=None, view=False, log=None):
            self.reg_methode = reg_method
            self.reg_method = reg_method
            self.source = source
            self.target = target
            self.out = out
            self.points_only = points_only
            self.t0 = t0
            self.r0 = r0
            self.s0 = s0
            self.batch = batch
            self.outdir = outdir
            self.outext = outext
            self.view = view
            self.log = log
    # Create Command line argument
    #command = 'gias-rigidreg corr_r -b rigidreg_list.txt -d aligned_meshes --outext .ply'


    #try to pass class in main function
    command = input(reg_method='corr_rs', batch='04_templates/Healthy_SSM.txt', outdir='08_healthy_ssm/00_aligned', out='.ply',)
    # different approach to subprocces 
    giasrigidreg.main(command)
    #giasrigidreg.main(parser.parse_args(command))
    # Run the command
    #subprocess.run(command, shell=True)   
    l = 1
    if l == 1:
        # create .txt file for fitted meshes 
        pca_list = os.listdir(os.path.join( ssm_path, '08_healthy_ssm/00_aligned'))

        # Save the list of text_file strings to a text file
        text_files_file_path = os.path.join(ssm_path, '04_templates/proximal_femur_correspondance_align.txt')
        with open(text_files_file_path, 'w') as text_files_file:
            for text_file in pca_list:
                text_file = os.path.join('08_healthy_ssm/00_aligned/', text_file)
                text_files_file.write(text_file + '\n')
#performs pca and creates shapemodel
if trainpcashapemodel == 1: 
    #just checking current directory
    os.chdir(ssm_path)
    path_file = '04_templates/Healthy_SSM_alpha_align.txt'
    #create class to parse input
    class PCA_input():
        def __init__(self, path_file=None, ncomponents=9, pc_path=None, points_only=False, recon_modes = [0, 1, 2, 3, 4, 5, 6, 7, 8], out='08_healthy_ssm/01_SSM_Model/femur_alpha', 
                     colour=[0.85, 0.8, 0.5], recon_colour_1=[0.9, 0.5, 0.5],
                     recon_colour_2= [0.5, 0.5, 0.9], bgcolour = [1.0, 1.0, 1.0], recon_opacity= 0.5, view=True, plot_pcs = 9, log=None ):
            self.path_file = path_file
            self.ncomponents = ncomponents
            self.pc_path = pc_path
            self.points_only = points_only
            self.recon_modes = recon_modes
            self.out = out
            self.colour = colour
            self.recon_colour_1 = recon_colour_1
            self.recon_colour_2 = recon_colour_2
            self.bgcolour = bgcolour
            self.recon_opacity = recon_opacity
            self.view = view
            self.plot_pcs = plot_pcs
            self.log = log            
    #add path_file to object
    pca_input = PCA_input(path_file=path_file)
    #run pca and create ssm
    gtrainpca.main(pca_input)
    # Create Command line argument
    #command = 'gias-trainpcashapemodel pca_list.txt -n 9 -r 0 1 2 -o shape_model/femur --plot_pcs 9 -v'

    # Run the command
    #subprocess.run(command, shell=True)   
#remeshing to reduce data points, laplacian filter to smoothen surface, flip around x axis 
if grooming == 1: 
    files = os.listdir(os.path.join( ssm_path, 'femur_pca_aligned'))
    for file in files:
        file_path = os.path.join(os.path.join( ssm_path, 'femur_pca_aligned/')+ file)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file_path)
        #ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.AbsoluteValue(1))
        #ms.apply_coord_laplacian_smoothing(stepsmoothnum=3)
        outfile = os.path.join(os.path.join( ssm_path, 'femur_pca_aligned/')+ file)
        ms.save_current_mesh(outfile)
    '''
    for file in files:
        if "L" in file and "F_" in file:
            file_path = os.path.join(os.path.join( ssm_path, 'segmentations/segmentations_femur/')+ file)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(file_path)
            ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.AbsoluteValue(2))
            ms.apply_coord_laplacian_smoothing(stepsmoothnum=1)
            ms.apply_matrix_flip_or_swap_axis(flipx=True)
            ms.meshing_invert_face_orientation()
            outfile = os.path.join(os.path.join( ssm_path, 'segmentations/segmentations_femur_female/')+ file)
            ms.save_current_mesh(outfile)
        elif "F_" in file and "R" in file:
            file_path = os.path.join(os.path.join( ssm_path, 'segmentations/segmentations_femur/')+ file)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(file_path)
            ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.AbsoluteValue(2))
            ms.apply_coord_laplacian_smoothing(stepsmoothnum=1)
            outfile = os.path.join(os.path.join( ssm_path, 'segmentations/segmentations_femur_female/')+ file)
            ms.save_current_mesh(outfile)
    
            # get principle_axes of spartial segmentation and transform
    '''
if init_alignment ==1:
    # different paths which need to be adapted
    no_cam_folder = "00_Data/05_Proximal_Femur_no_Cam"
    cam_folder = "00_Data/06_Proximal_Femur"
    aligned_cam_folder = "00_Data/06_Proximal_Femur"
    aligned_no_cam_folder = "00_Data/05_Proximal_Femur_no_Cam"
    shape_model_path = "03_No_Cam_SSM/femur_mean.ply"

    no_cam_trans = 1
    os.chdir(ssm_path)
    files = os.listdir(no_cam_folder)#
    files_Cam = os.listdir(cam_folder)#
    def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result


    def preprocess_point_cloud(pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def draw_registration_result(source, target, transformation):

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        import copy 
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([0.5, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])

        source_temp.transform(transformation)
        vis.add_geometry(target_temp)
        vis.add_geometry(source_temp)

        # visualize the aligned point cloud and target mesh
        vis.run()
        vis.destroy_window()
    #target file --> mean femur
    target_pcd = o3d.io.read_point_cloud(shape_model_path)
    target_centroid = vtktools.loadpoly(shape_model_path)
    target_centroid = np.mean(target_centroid.v, axis=0)
    for f in files:
        # get file name of Cam Femur
        try:
            f_cam = f #.replace('_noCam', '')
        except:
            print("CAM")
        cam_file = os.path.join(cam_folder, f_cam)#
        #include if statement to only use no cam but then transmatrix on cam
        s_file = os.path.join(no_cam_folder,f)#
        source_pcd = o3d.io.read_point_cloud(s_file)
        source_centroid = vtktools.loadpoly(s_file)
        source_centroid = np.mean(source_centroid.v, axis=0)
        init_transform = np.identity(4)
        init_transform[:3,3] = (source_centroid - target_centroid) * (-1)
        voxel_size= 5#.75
        threshold = 10
        print("Downsample the point cloud with a voxel of %0.1f" %voxel_size)
        source_downpcd, source_pcd_fpfh = preprocess_point_cloud(source_pcd, voxel_size=voxel_size)

        target_downpcd, target_pcd_fpfh = preprocess_point_cloud(target_pcd, voxel_size)

        #draw_registration_result(source_downpcd, target_downpcd, np.eye(4))
        # include ransac for initial alignment
        result = execute_global_registration(source_downpcd, target_downpcd, source_pcd_fpfh,
                                target_pcd_fpfh, voxel_size)
        
        #draw_registration_result(source_downpcd, target_downpcd, result.transformation)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        #draw_registration_result(source_pcd, target_pcd, reg_p2p.transformation)
        source_pcd = o3d.io.read_triangle_mesh(cam_file)
        source_pcd.transform(reg_p2p.transformation)
        source_pcd.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join(aligned_cam_folder, f_cam),source_pcd)
        if no_cam_trans == 1:
            source_pcd = o3d.io.read_triangle_mesh(s_file)
            source_pcd.transform(reg_p2p.transformation)
            source_pcd.compute_vertex_normals()
            o3d.io.write_triangle_mesh(os.path.join(aligned_no_cam_folder, f),source_pcd)



    # create .txt file for fitted meshes 
    pca_list = os.listdir(os.path.join( ssm_path, aligned_cam_folder))#
    if no_cam_trans == 1:
        no_cam_list = os.listdir(os.path.join( ssm_path, aligned_no_cam_folder))#

    # Save the list of text_file strings to a text file
    text_files_file_path = os.path.join(ssm_path, 'proximal_femur_no_cam_area_removal.txt')#
    with open(text_files_file_path, 'w') as text_files_file:
        for text_file in pca_list:
            text_file = os.path.join(aligned_cam_folder, text_file)#
            text_files_file.write(text_file + '\n')
    text_file_path_noCam = os.path.join(ssm_path, 'proximal_femur_cam_area_removal.txt')
    if no_cam_trans == 1:
        with open(text_file_path_noCam, 'w') as text_files_file_noCam:    
            for text_file in no_cam_list:
                text_file = os.path.join(aligned_no_cam_folder, text_file)#
                text_files_file_noCam.write(text_file + '\n')
# fit ssm to sparse data
if fit_shapemodel == 1: 
    os.chdir(ssm_path)
    class pcareg():
        def __init__(self, mean, ssm, fit_comps=[0, 1, 2, 3, 4, 5, 6, 7, 8], fit_mode='ts', target=None, out=None, auto_align=True, 
                      mweight=0.2, sample = None, fit_scale=False, points_only=None, batch=None, outdir=None,
                       outext='.ply', view=False, log=None, out_csv=None):
            self.mean = mean
            self.ssm = ssm
            self.fit_comps = fit_comps
            self.fit_mode = fit_mode
            self.target = target
            self.out = out
            self.auto_align = auto_align
            self.mweight = mweight
            self.sample = sample
            self.fit_scale = fit_scale
            self.points_only = points_only
            self.batch = batch
            self.outdir = outdir
            self.outext = outext
            self.view = view
            self.log = log
            self.out_csv = out_csv
    
    mean_shape = 'F:/Public/14_Femur_Griffith/08_healthy_ssm/01_SSM_Model/femur_alpha_mean.ply' # '03_SSM/00_Healthy_SSM/femur_mean.ply'
    ssm = 'F:/Public/14_Femur_Griffith/08_healthy_ssm/01_SSM_Model/femur_alpha.pc.npz' #'03_SSM/00_Healthy_SSM/femur.pc.npz'
    fit_comps = [0, 1, 2, 3, 4, 5, 6, 7] #, 8, 9, 10, 11, 12, 13, 14, 15]
    target = 'F:/Public/00Data/Models/Subject_05/Bones/Landmarks/Right_Femur_ISB_cut_align.ply' #'00_Data/07_Proximal_Femur_no_Cam_pcfit/'
    batch = '01_txt_files/No_cam_Data.txt'
    out = os.path.splitext(target)[0] + '_recon.ply' # maybe automize it 
    outdir = '00_Data/07_Proximal_Femur_no_Cam_pc_fit' #os.path.join(ssm_path, 'pc_recon_no_cam/')
    csv_output = "F:/Public/12_Femur_VIFM/00_Data/Fit_VIFM.csv"
        # add alignment step before putting it into pca registration


    pc_reg_input = pcareg(mean=mean_shape, ssm=ssm, fit_comps=fit_comps, fit_mode='ts', out=out,  outdir=outdir, auto_align=True, mweight=0.01, target=None,  batch = batch, out_csv=csv_output)
    pcreg.main(pc_reg_input)
        
# calculate surface to surface distance 
if surface_to_surface == 1: 
    
    os.chdir(r"D:\Public\AUT\SSM")
    # create object to pass input parameters in function
    class stos():
        def __init__(self, gTruthPath, testPath, outpath=None, display=False, groundtruth_unit='mm', test_unit='mm'):
            self.gTruthPath = gTruthPath
            self.testPath = testPath
            self.outpath = outpath
            self.display = display
            self.groundtruth_unit = groundtruth_unit
            self.test_unit = test_unit
    #define paths where files are located 
    gTruthpath = 'segmentations_partial/Sub_009_Femur_L_proc_noCam_ssmreg.ply' #get acutal ply file out of if
    testPath =   'segmentations_partial/Sub_009_Femur_L_proc_ssmreg.ply'
    outpath = 'segmentations_partial/results.txt'
    surface_input = stos(gTruthpath, testPath, outpath=outpath)
    
    gsurf.main(surface_input)


if nearest_neighbor_distance == 1: 
    target_fem = 'shape_model_normal/pelvis_mean.ply'
    original_fem = 'shape_model_mild/pelvis_mean.ply'
    #using pyvista to read ply files
    mesh_recon = pv.read(original_fem)
    target_mesh = pv.read(target_fem)
    #creating KDTree 
    tree = KDTree(target_mesh.points) # heretarget mesh
    d_kdtree, idx = tree.query(mesh_recon.points) #here mesh recon

    #creating distances
    mesh_recon["distances"] = d_kdtree #here mesh recon
    #compute mean distance
    print(np.mean(d_kdtree))

    #visualize results
    p = pv.Plotter()
    p.add_mesh(mesh_recon, scalars="distances", cmap="rainbow", clim=[np.min(d_kdtree), 3],
           smooth_shading=True, scalar_bar_args={"title": "Distances"})
    #p.add_mesh(target_mesh, color=True, opacity=0.5, smooth_shading=True)
    p.show()

if get_point_id_list_vtk == 1: 

    # Load the femur and cam meshes using vtkPLYReader
    femur_path = "00_Data/02_Radial_Basis_Fit/20-29_F_16_3362_Fem_L_2mm_flipped_rbfreg.ply"
    cam_path = "02_templates/femur_head_metrics.ply"

    # Create vtkPLYReader objects to load the meshes
    femur_reader = vtk.vtkPLYReader()
    femur_reader.SetFileName(femur_path)
    femur_reader.Update()  # Load the femur mesh
    femur_polydata = femur_reader.GetOutput()

    cam_reader = vtk.vtkPLYReader()
    cam_reader.SetFileName(cam_path)
    cam_reader.Update()  # Load the cam mesh
    cam_polydata = cam_reader.GetOutput()

    # Extract points from femur and cam meshes
    points_femur = femur_polydata.GetPoints()
    points_cam = cam_polydata.GetPoints()

    # Convert points to numpy arrays for easy manipulation
    num_points_femur = points_femur.GetNumberOfPoints()
    num_points_cam = points_cam.GetNumberOfPoints()

    # Round and store the femur and cam points
    points_femur_np = np.round(np.array([points_femur.GetPoint(i) for i in range(num_points_femur)]), decimals=2)
    points_cam_np = np.round(np.array([points_cam.GetPoint(n) for n in range(num_points_cam)]), decimals=2)

    print(points_femur_np[3619])  # Print specific point for debugging

    # Use vtkPointLocator to find the nearest point in the cam mesh for each femur point
    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(cam_polydata)
    point_locator.BuildLocator()

    # Iterate through femur points and find corresponding points in cam
    Point_List_ID = []
    for i, coord_fem in enumerate(points_femur_np):
        # Find the nearest point in the cam mesh
        closest_point_id = point_locator.FindClosestPoint(coord_fem)
        closest_point = points_cam_np[closest_point_id]

        # Check if the points are exactly the same
        if np.all(coord_fem == closest_point):
            Point_List_ID.append(i)

    # Save the matching points
    points_to_save = np.array(Point_List_ID)
    np.savetxt('01_txt_files/femur_head_metrics.txt', points_to_save)


if get_point_id_list == 1: 

        # Load the femur and cam meshes
    femur = "00_Data/02_Radial_Basis_Fit/20-29_F_16_3362_Fem_L_2mm_flipped_rbfreg.ply"
    cam = "02_templates/lateral_condyle_elip_metrics.ply"
    VTKMeshUtl = data.VTKMeshUtl.load(femur)
    VTKMeshUtlCam = data.VTKMeshUtl.load(cam)

    # Extract points from the meshes
    num_points = VTKMeshUtl.GetNumberOfPoints()
    num_points_cam = VTKMeshUtlCam.GetNumberOfPoints()
    points_femur = [VTKMeshUtl.GetPoint(i) for i in range(num_points)]
    points_cam = [VTKMeshUtlCam.GetPoint(n) for n in range(num_points_cam)]

    # Convert points to numpy arrays and round them
    points_femur = np.round(np.array(points_femur), decimals=2)
    points_cam = np.round(np.array(points_cam), decimals=2)


    # Build a KDTree for the cam points
    kd_tree_cam = KDTree(points_cam)

    # Query the KDTree with femur points to find matches
    Point_List_ID = []
    for i, coord_Fem in enumerate(points_femur):
        distances, indices = kd_tree_cam.query(coord_Fem, k=1)  # k=1 for nearest neighbor
        
        if np.all(coord_Fem == points_cam[indices]):
            Point_List_ID.append(i)

    # Save the matching points
    points = np.array(Point_List_ID)
    np.savetxt('01_txt_files/lateral_condyle_elip_metrics.txt', points)
    '''
    femur = "00_Data/03_PC_aligned/20-29_F_16_3075_Fem_L_2mm_flipped_rbfreg.ply"
    cam = "02_templates/distal_femur_template.ply"
    VTKMeshUtl = data.VTKMeshUtl.load(femur)
    num_points = VTKMeshUtl.GetNumberOfPoints()
    VTKMeshUtlCam = data.VTKMeshUtl.load(cam)
    num_points_cam = VTKMeshUtlCam.GetNumberOfPoints()
    Point_List_ID = []
    for i in range(num_points):
        for n in range(num_points_cam):
            coord_Fem = np.array(VTKMeshUtl.GetPoint(i))
            coord_Cam = np.array(VTKMeshUtlCam.GetPoint(n))
            
            coord_Fem = np.round(coord_Fem, decimals=2)
            coord_Cam = np.round(coord_Cam, decimals=2)
            result = (coord_Fem == coord_Cam).all()

            if result == True:
                Point_List_ID.append(i)   
    points = np.array(Point_List_ID)
    np.savetxt('01_txt_files/template_cut_to_proximal_vifm.txt', points)'''

if cut_out_cam == 1:
    femur_list = os.listdir("00_Data/08_Proximal_Femur_Alpha_align")
    femur_folder = "00_Data/08_Proximal_Femur_Alpha_align"
    point_ids = np.loadtxt("01_txt_files/cam_area_template.txt")
    point_ids = point_ids.tolist()
    for femur in femur_list:
        VTKMeshUtl = data.VTKMeshUtl.load(os.path.join(femur_folder, femur))
        #get number of Points to iterate over it later
        num_points = VTKMeshUtl.GetNumberOfPoints()
        Points_sublist = []
        for i in range(num_points):
            if i in point_ids:
                Points_sublist.append(i)
        #Create submesh without CAM area
        Cut_Mesh = data.VTKMeshUtl.sub_mesh(Points_sublist, VTKMeshUtl)
        femur_new = (femur[:21] + '_prox_noCam.ply')
        femur_new = ('F:/Public/12_Femur_VIFM/00_Data/05_Proximal_Femur_no_Cam/' + femur_new)
        # write cut mesh in new_Folder
        data.VTKMeshUtl.write(femur_new, Cut_Mesh)

if cut_out_acetabulum == 1:
    femur_list = os.listdir("femur_transformed")
    femur_folder = "femur_transformed"
    point_ids = np.loadtxt("D:/Public/07_CamSSM/pelvis/points_prox_femur.txt")
    point_ids = point_ids.tolist()
    for femur in femur_list:
        VTKMeshUtl = data.VTKMeshUtl.load(os.path.join(femur_folder, femur))
        #get number of Points to iterate over it later
        num_points = VTKMeshUtl.GetNumberOfPoints()
        Points_sublist = []
        for i in range(num_points):
            if i not in point_ids:
                Points_sublist.append(i)
        #Create submesh without CAM area
        Cut_Mesh = data.VTKMeshUtl.sub_mesh(Points_sublist, VTKMeshUtl)
        femur_new = (os.path.splitext(femur)[0] + '_prox.ply')
        femur_new = ('D:/Public/07_CamSSM/pelvis/femur_proximal/' + femur_new)
        # write cut mesh in new_Folder
        data.VTKMeshUtl.write(femur_new, Cut_Mesh)

# here the cam area of the actual segmentation is compared to a healthy SSM fitted to the segmentation where cam area was cut out
if define_CAM_state == 1: 
    visual = 0
    segmented_dir = "F:/Public/12_Femur_VIFM/00_Data/08_Proximal_Femur_Alpha_align"
    recon_dir = "F:/Public/12_Femur_VIFM/00_Data/07_Proximal_Femur_no_Cam_pc_fit"
    segmented_files = os.listdir(segmented_dir)
    recon_files = os.listdir(recon_dir)
    point_ids = np.loadtxt("F:/Public/12_Femur_VIFM/01_txt_files/cam_area_template2.txt")
    columns = [ "subject", "avg. distance", "max. distance", "min. distance"]
    avg_list = []
    max_list = []
    min_list = []
    volume_diff_list = []

    # Helper function to triangulate the mesh
    def triangulate(polydata):
        triangulate_filter = vtk.vtkTriangleFilter()
        triangulate_filter.SetInputData(polydata)
        triangulate_filter.Update()
        return triangulate_filter.GetOutput()

    # Helper function to calculate volume between two surfaces
    def calculate_volume_between_surfaces(polydata_b, polydata_a, point_ids):
        signed_distance_filter = vtk.vtkImplicitPolyDataDistance()
        signed_distance_filter.SetInput(polydata_b)

        volume = 0.0

        for point_id in point_ids:
            point_a = np.array(polydata_a.GetPoint(int(point_id)))
            point_b = np.array(polydata_b.GetPoint(int(point_id)))
            
            # Compute the signed distance
            signed_distance = signed_distance_filter.EvaluateFunction(point_a)
            
            # Compute the average area of the surrounding triangles
            cells = vtk.vtkIdList()
            polydata_a.GetPointCells(int(point_id), cells)
            
            avg_area = 0.0
            if signed_distance > 0:
                for i in range(cells.GetNumberOfIds()):
                    cell = polydata_a.GetCell(cells.GetId(i))
                    pts = cell.GetPoints()
                    if pts.GetNumberOfPoints() == 3:
                        p0 = np.array(pts.GetPoint(0))
                        p1 = np.array(pts.GetPoint(1))
                        p2 = np.array(pts.GetPoint(2))
                        area = np.linalg.norm(np.cross(p1 - p0, p2 - p0)) / 2.0
                        avg_area += area
                
                avg_area /= cells.GetNumberOfIds()
                
                # Volume contribution by this point
                volume += signed_distance * avg_area

        return abs(volume)
    #cam_points = cam_points.tolist()
    for i in range(len(recon_files)):
        # Load Mesh A
        reader_a = vtk.vtkPLYReader()
        reader_a.SetFileName(os.path.join(recon_dir, recon_files[i]))
        reader_a.Update()  # Force update to retrieve the points
        polydata_a = reader_a.GetOutput() #recon dir is inputed 
        rec_file = recon_files[i]
        print(rec_file[:25])
        seg_file = [segmentation for segmentation in segmented_files if rec_file[:25] in segmentation]
        # Load Mesh B
        reader_b = vtk.vtkPLYReader()
        reader_b.SetFileName(os.path.join(segmented_dir, seg_file[0]))
        reader_b.Update()  # Force update to retrieve the points
        polydata_b = reader_b.GetOutput()
        
        test_seg =seg_file[0]
        if rec_file[:20] != test_seg[:20]:
            print("Error! Files are not the same!")
            
        # Load Point List (assuming point_list contains point IDs)
        #point_ids = np.array(your_point_list)
        
        # Create a signed distance filter (of the reconstructed file)
        signed_distance_filter = vtk.vtkImplicitPolyDataDistance()
        signed_distance_filter.SetInput(polydata_a)

        # Calculate Signed Distances
        signed_distances = []
        divider = 0
        for point_id in point_ids:
            # Get the coordinates for the current point ID in Mesh A
            point_b = polydata_b.GetPoint(int(point_id))

            # Evaluate the signed distance at the point on Mesh A
            signed_distance = signed_distance_filter.EvaluateFunction(point_b)
            '''if signed_distance < 0:
                signed_distance = 0
                
            if signed_distance > 0:
                divider = divider + 1'''
            signed_distances.append(signed_distance)
        def get_highest_lowest_value_average(arr, arg, N):
            sorted_arr = sorted(arr, reverse=arg)
            top_N = sorted_arr[:N]
            return np.mean(top_N)
        #calculate statistics 
        max_diff = get_highest_lowest_value_average(signed_distances, True, 5) #np.max(np.array(signed_distances))
        min_diff = get_highest_lowest_value_average(signed_distances, False, 5)
        avg_diff = np.mean(signed_distances) #/ divider 

        # Append data to the DataFrame
        avg_list.append(avg_diff)
        max_list.append(max_diff)
        min_list.append(min_diff)

        # Calculate the volume between the two surfaces in the area defined by point_ids
        volume_diff = calculate_volume_between_surfaces(polydata_a, polydata_b, point_ids) # why is a here first? 
        volume_diff_list.append(volume_diff)
        enum_i = i

        if visual == 1:
            # Visualization
            mapper_b = vtk.vtkPolyDataMapper()
            mapper_b.SetInputData(polydata_b)
            mapper_b.SetScalarModeToUseCellData() 

            # Create an actor for Mesh B
            actor_b = vtk.vtkActor()
            actor_b.SetMapper(mapper_b)
            # Set default color (grey) for Mesh B
            actor_b.GetProperty().SetColor(0.5, 0.5, 0.5)


            # Create a color transfer function for custom coloring
            color_transfer_function = vtk.vtkColorTransferFunction()
            color_transfer_function.AddRGBPoint(np.min(signed_distances), 0.5, 0.5, 0.5)  # Grey for values below 0
            color_transfer_function.AddRGBPoint(0.0, 0.0, 0.0, 1.0)  # Blue for 0
            color_transfer_function.AddRGBPoint(np.max(signed_distances), 1.0, 0.0, 0.0)  # Red for maximum # Blue for maximum

            # Set the color transfer function for the mapper
            mapper_b.SetLookupTable(color_transfer_function)

            # Create a renderer and render window
            renderer = vtk.vtkRenderer()
            render_window = vtk.vtkRenderWindow()
            render_window.SetWindowName("VTK Visualization")
            render_window.SetSize(800, 600)
            render_window.AddRenderer(renderer)

            # Create a render window interactor
            render_window_interactor = vtk.vtkRenderWindowInteractor()
            render_window_interactor.SetRenderWindow(render_window)

            # Add the actor for Mesh B to the renderer
            renderer.AddActor(actor_b)

            # Create spheres to represent the points where signed distances were calculated
            point_size = 0.05
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetRadius(point_size)

            # Add color to Mesh B based on signed distances
            color_scalars = vtk.vtkDoubleArray()
            color_scalars.SetNumberOfValues(polydata_b.GetNumberOfCells())
            num_Cells = polydata_b.GetNumberOfCells()
            # Set default color (e.g., grey) to all cells
            default_color = np.min(signed_distances) # Adjust the default color value as needed
            color_scalars.FillComponent(0, default_color)


            for i, signed_distance in enumerate(signed_distances):
                point_idx = int(point_ids[i])

                # Get the connected cells to the point
                cell_ids = vtk.vtkIdList()
                polydata_b.GetPointCells(point_idx, cell_ids)
                #polydata_b.GetCellNeighbors(-1, point_idx, cell_ids)

                # Set the signed_distance value for each connected cell
                for j in range(cell_ids.GetNumberOfIds()):
                    cell_id = cell_ids.GetId(j)
                    color_scalars.SetValue(cell_id, signed_distance)
                    
                
            
            polydata_b.GetCellData().SetScalars(color_scalars)

            # Update the color range
            mapper_b.SetScalarRange(np.min(signed_distances), np.max(signed_distances))

            # Set up the camera
            renderer.ResetCamera()
            renderer.GetActiveCamera().Azimuth(30)
            renderer.GetActiveCamera().Elevation(30)
            renderer.GetActiveCamera().Dolly(1.2)

            # Set the background color
            renderer.SetBackground(0.5, 0.5, 0.5)

                        # Create a color transfer function for custom coloring
            color_transfer_function = vtk.vtkColorTransferFunction()
            color_transfer_function.AddRGBPoint(np.min(signed_distances), 0.5, 0.5, 0.5)  # Grey for values below 0
            color_transfer_function.AddRGBPoint(0.0, 0.0, 0.0, 1.0)  # Blue for 0
            color_transfer_function.AddRGBPoint(np.max(signed_distances), 1.0, 0.0, 0.0)  # Red for maximum # Blue for maximum

            # Set the color transfer function for the mapper
            mapper_b.SetLookupTable(color_transfer_function)

            # Create a scalar bar actor
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(color_transfer_function)
            scalar_bar.SetTitle("Signed Distances in (mm)")
            scalar_bar.SetNumberOfLabels(4)  # Adjust the number of labels as needed
            scalar_bar.SetPosition(0.85, 0.1)
            scalar_bar.SetWidth(0.1)
            scalar_bar.SetHeight(0.8)
            
            scalar_bar.SetUnconstrainedFontSize(True)
            

            # Add a text actor to display the segmented file name
            text_actor = vtk.vtkTextActor()
            text_actor.SetTextScaleModeToNone()
            text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
            text_actor.SetPosition(0.1, 0.9)  # Adjust position as needed
            text_actor.GetTextProperty().SetFontSize(18)
            text_actor.GetTextProperty().SetColor(0, 0, 0)  # Text color (black)
            #text_actor.SetInput("Segmented File: " + segmented_files[enum_i] + "   Reconstructed File" + recon_files[enum_i])


            # Add the actor for Mesh B to the renderer
            renderer.AddActor(actor_b)
            renderer.AddActor(scalar_bar) 
            renderer.AddActor(text_actor)
            # Render the scene
            render_window.Render()

            # Start the interactor
            render_window_interactor.Start()

    df = pd.DataFrame({'Subject': recon_files, 'Average Distance': avg_list, 'Maximal Distance': max_list, 'Minimal Distance': min_list, 'Volume': volume_diff_list})
    os.chdir(r"F:\Public\12_Femur_VIFM")
    df.to_csv("Cam_Volume_Calculation.csv")
    
# this script takes acetabulum segmentation, fits sphere to it and translates it towards the center of the sphere,
# then it takes whole femur, cuts out femur head, fits spehere to it and transforms it into the isb coordinate system
# finally it gets acetabulum and proximal femur, flattens them to have them in same mesh 
if center_femur_to_FHC:
    spehere_fit_meshes = os.listdir("femur_proximal_train/cup")
    femur_meshes = os.listdir("femur_proximal_train/femur_prox")
    for i in range(len(femur_meshes)): 
        femur_mesh = o3d.io.read_triangle_mesh(os.path.join("femur_proximal_train/femur_prox/", femur_meshes[i]))
        sphere_fit_mesh = o3d.io.read_point_cloud(os.path.join("femur_proximal_train/cup/", spehere_fit_meshes[i]))
        sphere_fit_points = np.asarray(sphere_fit_mesh.points)
        center = trans_sphere.Cloud.sphere_fit(sphere_fit_points)
        # create transformation matrix 
        trans_matrix = np.identity(4)
        trans_matrix[0:3,3] = (-1) * center.reshape(-1,3)
        femur_mesh.transform(trans_matrix)
        femur_mesh.compute_vertex_normals
        o3d.io.write_triangle_mesh(os.path.join("femur_proximal_griffith_saarbruecken", femur_meshes[i]), femur_mesh)

if get_coupled_mesh == 1: 
    sphere_fit_meshes = os.listdir("sphere_fit")
    acetabulum_meshes = os.listdir("acetabulum")
    hemi_pelvis = os.listdir("fitted_meshes")
    sphere_fit_femur_meshes = os.listdir("femur_cup")
    femur_meshes = os.listdir("femur")
    prox_femur_meshes = os.listdir("femur_proximal")

    for i in range(len(sphere_fit_meshes)):
        sphere_fit_mesh = o3d.io.read_point_cloud(os.path.join("sphere_fit", sphere_fit_meshes[i]))
        # extract points from mesh
        sphere_fit_points = np.asarray(sphere_fit_mesh.points)
        #sphere_fit_points = sphere_fit_points.reshape(-1, 3)
        # use library from ted to get center
        center = trans_sphere.Cloud.sphere_fit(sphere_fit_points)
        # create transformation matrix 
        trans_matrix = np.identity(4)
        trans_matrix[0:3,3] = (-1) * center.reshape(-1,3)
        #now read in acetabulum
        mesh_acetabulum = o3d.io.read_triangle_mesh(os.path.join("acetabulum", acetabulum_meshes[i])) #copy.deepcopy(source).transform(reg_p2p.transformation)
        mesh_acetabulum.transform(trans_matrix) 
        mesh_acetabulum.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join("acetabulum_transformed", acetabulum_meshes[i]), mesh_acetabulum)
        # apply transform to hemipelvis 
        mesh_hemi_pelvis = o3d.io.read_triangle_mesh(os.path.join("fitted_meshes", hemi_pelvis[i])) #copy.deepcopy(source).transform(reg_p2p.transformation)
        mesh_hemi_pelvis.transform(trans_matrix) 
        mesh_hemi_pelvis.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join("hemi_pelvis_transformed", hemi_pelvis[i]), mesh_hemi_pelvis)
        # now do same for femur 
        sphere_fit_femur_mesh = o3d.io.read_point_cloud(os.path.join("femur_cup", sphere_fit_femur_meshes[i]))
        sphere_fit_femur_points = np.asarray(sphere_fit_femur_mesh.points)

        center = trans_sphere.Cloud.sphere_fit(sphere_fit_femur_points)

        # load complete femur 
        femur_mesh = o3d.io.read_point_cloud(os.path.join("femur", femur_meshes[i]))


        VTKMeshUtl = data.VTKMeshUtl.load(os.path.join("femur", femur_meshes[i]))
        #extract point on top of femur head
        knee_lat = np.array(VTKMeshUtl.GetPoint(17842))
        #extract two points on femur neck to shaft connection
        knee_med = np.array(VTKMeshUtl.GetPoint(2938))
        #now get isb coord
        knee_mid = (knee_lat + knee_med)/2

        y_axis = center.reshape(1,3) - knee_mid.reshape(1,3)
        z_axis_temp = knee_lat.reshape(1,3) - knee_med.reshape(1,3)
        x_axis = np.cross(y_axis, z_axis_temp)

        z_axis = np.cross(x_axis, y_axis)

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        transform_matrix = np.identity(4)
        rot_matrix = np.vstack((z_axis, x_axis, y_axis))
        transform_matrix[:3,:3] = rot_matrix

        #transform_matrix[:3,3] = (1) * center.reshape(-1,3)
        print(transform_matrix)
        femur_mesh = o3d.io.read_triangle_mesh(os.path.join("femur", femur_meshes[i]))
        femur_mesh.transform(transform_matrix)
        femur_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join("femur_transformed", femur_meshes[i]), femur_mesh)
        # read rotatet femur in again 
        VTKMeshUtl = data.VTKMeshUtl.load(os.path.join("femur_transformed", femur_meshes[i]))

        point_ids = np.loadtxt("D:/Public/07_CamSSM/pelvis/points_sphere_fit_femur.txt")
        point_ids = point_ids.astype(int)
        # Initialize a list to store the locations of each vertex
        vertex_locations = []

        for point in point_ids:
            vertex_location = np.array(VTKMeshUtl.GetPoint(point))
            vertex_locations.append(vertex_location)
        # Calculate new femur head location
        center = trans_sphere.Cloud.sphere_fit(np.asarray(vertex_locations))

        sphere_transformation_matrix = np.identity(4)
        sphere_transformation_matrix[:3,3] = -center.reshape(-1,3)

        femur_mesh = o3d.io.read_triangle_mesh(os.path.join("femur_transformed", femur_meshes[i]))
        femur_mesh.transform(sphere_transformation_matrix)
        femur_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join("femur_transformed", femur_meshes[i]), femur_mesh)
        print("It worked")
        #read in proximal femur
        prox_femur_mesh = o3d.io.read_triangle_mesh(os.path.join("femur_proximal", prox_femur_meshes[i]))
        
        # to do 
        combined_mesh = o3d.geometry.TriangleMesh()
        # Append vertices and triangles from the first mesh (femur_mesh)
        combined_mesh.vertices = prox_femur_mesh.vertices
        combined_mesh.triangles = prox_femur_mesh.triangles

        # Get the number of vertices in the first mesh for indexing the second mesh
        num_vertices_femur = np.asarray(prox_femur_mesh.vertices).shape[0]
        #num_vertices_femur = len(num_vertices_femur)

       # Append vertices and triangles from the second mesh (mesh_hemi_pelvis)
        combined_mesh.vertices = o3d.utility.Vector3dVector(
            np.concatenate([np.asarray(combined_mesh.vertices), np.asarray(mesh_acetabulum.vertices)])
        )
        combined_mesh.triangles = o3d.utility.Vector3iVector(
            np.concatenate([np.asarray(combined_mesh.triangles), np.asarray(mesh_acetabulum.triangles) + num_vertices_femur])
        )
        # Compute normals for the combined mesh
        combined_mesh.compute_vertex_normals()
        output_name = prox_femur_meshes[i]
        output_name = output_name[0:16] + "Hip_Joint" + output_name[19:21] + ".ply"
        # Save the combined mesh to a file
        o3d.io.write_triangle_mesh(os.path.join("hip_joint", output_name), combined_mesh)
        # flatten meshes prox and acetabulum

if get_femur_measurements == 1:
    print_all_landmarks = 1
    femur_path = '00_Data/02_Radial_Basis_Fit'
    femur_list = os.listdir(femur_path)
 
    transform_to_center = 1
    femur_measurements = pd.DataFrame(data = None, index = femur_list, 
                                      columns= ["CCD_Angle", "Anterversion_Angle", "Femur_Length", 
                                                "Femur_Head_Diameter", "Neck_Length" ])
                                                                                 
    fix_orientations = np.array([-1, -1 , 1])
    for i in range(len(femur_list)):
        print(femur_list[i])
        # use data.VTKmehsload to load femur mesh
        femur_mesh = data.VTKMeshUtl.load(os.path.join(femur_path, femur_list[i]))
        if transform_to_center ==1: 
            # calculate diameter of femur head 
            point_ids_femur_head = np.loadtxt("01_txt_files/femur_head_metrics.txt")
            #use sphere fit to get diameter 
            diameter, center = sphere_fit(get_point_clound(femur_mesh, point_ids_femur_head))# )
            # Translate the mesh to center it at the origin
            transform = vtk.vtkTransform()
            transform.Translate(-center[0], -center[1], -center[2])

            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputData(femur_mesh)
            transform_filter.SetTransform(transform)
            transform_filter.Update()
            femur_mesh =transform_filter.GetOutput()
        # load point list shaft
        point_ids_femur_shaft_prox = np.loadtxt("01_txt_files/proximal_shaft_femur_metrics.txt")
        # get all coordinates of shaft using the template point id and calculate centroid 
        #get number of Points to iterate over it later
        
        centroid_shaft_prox = get_centroid(femur_mesh, point_ids_femur_shaft_prox) 
        
        point_ids_femur_shaft_distal = np.loadtxt("01_txt_files/distal_shaft_femur_metrics.txt")
        # get all coordinates of shaft using the template point id and calculate centroid 
        #get number of Points to iterate over it later
        centroid_shaft_dist= get_centroid(femur_mesh, point_ids_femur_shaft_distal) 

        # get all coordinates of neck using the template point id and calculate centroid
        point_ids_femur_neck = np.loadtxt("01_txt_files/femur_neck_metrics.txt")
        # get all coordinates of shaft using the template point id and calculate centroid 
        centroid_neck = get_centroid(femur_mesh, point_ids_femur_neck) 
        # calculate diameter of femur head 
        point_ids_femur_head = np.loadtxt("01_txt_files/femur_head_metrics.txt")
        # get all coordinates of shaft using the template point id and calculate centroid 
        
        #use sphere fit to get diameter 
        diameter, center = sphere_fit(get_point_clound(femur_mesh, point_ids_femur_head) )
        
        # calculate epicondyles using ellipsoid fit
        point_ids_femur_lateral_cond = np.loadtxt("01_txt_files/lateral_condyle_elip_metrics.txt") #make sure to create txt
        point_ids_femur_medial_cond = np.loadtxt("01_txt_files/medial_condyle_elip_metrics.txt")

        med_cond_pc = get_point_clound(femur_mesh, point_ids_femur_medial_cond) 
        lat_cond_pc = get_point_clound(femur_mesh, point_ids_femur_lateral_cond) 
        
        center_med, evecs, radii, v = ellipsoid_fit(med_cond_pc)
        center_lat, evecs, radii, v = ellipsoid_fit(lat_cond_pc)

        def femoral_anteversion_angle(center_med, center_lat, shaft_neck_intersect, femur_head, centroid_neck, lat_con, med_cond):
            # Create local coordinate system for femur where x_axis is from midpoint center_med/center_lat to shaft_neck_intersect
            x_axis = shaft_neck_intersect - (center_med + center_lat) / 2
            x_axis /= np.linalg.norm(x_axis)

            # Temporary y_axis goes from center_med to center_lat
            y_axis = center_lat - center_med
            y_axis /= np.linalg.norm(y_axis)

            # z_axis is cross product of x_axis and y_axis
            z_axis = np.cross(x_axis, y_axis)
            z_axis /= np.linalg.norm(z_axis)

            # y_axis is cross product of z_axis and x_axis to ensure orthogonality
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)

            # Save in local_transform_matrix
            local_transform_matrix = np.identity(4)
            local_transform_matrix[0, :3]  = x_axis
            local_transform_matrix[1, :3] = y_axis
            local_transform_matrix[2, :3] = z_axis
            print(local_transform_matrix)
            # Apply transformation to femur_head, centroid_neck, lat_cond, med_cond and project them on plane spanned by y_axis and z_axis
            femur_head_transformed = np.dot(local_transform_matrix[:3, :3], femur_head)
            centroid_neck_transformed = np.dot(local_transform_matrix[:3, :3], centroid_neck)
            lat_cond_transformed = np.dot(local_transform_matrix[:3, :3], lat_cond)
            med_cond_transformed = np.dot(local_transform_matrix[:3, :3], med_cond)

            # Project points onto the plane spanned by y_axis and z_axis (i.e., ignore x component)
            femur_head_proj = femur_head_transformed[1:]
            centroid_neck_proj = centroid_neck_transformed[1:]
            lat_cond_proj = lat_cond_transformed[1:]
            med_cond_proj = med_cond_transformed[1:]

            # Calculate angle between vector femur_head & centroid_neck and lat_cond & med_cond
            vector_neck_proj = centroid_neck_proj - femur_head_proj
            vector_cond_proj = lat_cond_proj - med_cond_proj

            femoral_anteversion_angle = angle_between_vectors(vector_neck_proj, vector_cond_proj)
            return femoral_anteversion_angle

        # build vector from 0,0,0 (Femur Head Center) to neck centroid to build neck vector#
        vector_neck = centroid_neck #prob delete, here i first need to find point on plane diaphsis_prox, 
        #dist femur head of which the vector point_plane , mid neck is perpendicular to plane

        
        # vector epicondyls
        # load only tip of condyles and calculate centroid
        point_ids_femur_lateral_cond_tip = np.loadtxt("01_txt_files/lateral_condyle_metrics.txt")
        point_ids_femur_medial_cond_tip = np.loadtxt("01_txt_files/medial_condyle_metrics.txt")
       

        #condyle1 = center_med #np.array(femur_mesh.GetPoint(2938))
        #condyle2 = center_lat #np.array(femur_mesh.GetPoint(17842))
        # build vector from mid_condyles to to centroid shaft
        vector_shaft = centroid_shaft_prox - centroid_shaft_dist 
         # extract condyle center
        lat_cond = get_centroid(femur_mesh, point_ids_femur_lateral_cond_tip)
        med_cond = get_centroid(femur_mesh, point_ids_femur_medial_cond_tip) 
        vector_epicondyles =   get_centroid(femur_mesh, point_ids_femur_lateral_cond_tip)  - get_centroid(femur_mesh, point_ids_femur_medial_cond_tip)  # 17714 13123 center_lat -center_med

        #test neck-shaft angle without point on plane
        neck_shaft_not_on_plane = angle_between_vectors(vector_neck, vector_shaft)

        # calculate angle between shaft vecotr and neck vector
        femur_head = np.array([0, 0, 0])
        neck_shaft_angle, neck_vector, point_on_plane = calculate_neck_shaft_angle(femur_head, centroid_shaft_prox, centroid_shaft_dist, centroid_neck)
        # test different angle calculation

        neck_shaft_angle_test = angle_between_vectors(vector_neck, vector_shaft)
        #plotter_vector(vector_neck[[0,1]], vector_epicondyles[[0,1]])
        # calculate angle between neck vector and epicondyle vector
        vectorneck = vector_neck[[0,1]]
        vectorepi = vector_epicondyles[[0,1]]

        neck_epi_angle = angle_between_vectors_2d(vectorneck, vectorepi)
        # calculate angle between neck vector and epicondyle vector

        femoral_anteversion = neck_epi_angle #angle_between_vectors(vector_neck, vector_epicondyles)
        if femoral_anteversion < 0:
            print(femur_list[i] + " has negative anteversion")
        
        unit_epi = vector_epicondyles / np.linalg.norm(vector_epicondyles)
        unit_neck = vector_neck / np.linalg.norm(vector_neck)
        zero_vec = np.array([0, 1])
        # calc slope
        neck_to_zero = angle_between_vectors(unit_neck[[0,1]], zero_vec)
        epi_to_zero = angle_between_vectors(unit_epi[[0,1]], zero_vec)
        # this ensures that angle can be negative
        femoral_anteversion = epi_to_zero - neck_to_zero 
        if femoral_anteversion < 0:
            print(femur_list[i] + " has negative anteversion")
                # calculate neck length by shifting the shaft vector so neck vector and shaft vector intersect, length = intersection point to femur cup center 
        # Find the intersection point
        P1 = femur_head
        P2 = point_on_plane
        P3 = centroid_shaft_dist
        P4 = centroid_shaft_prox
        Pa_optimal, Pb_optimal = shortest_line_segment(P1, P2, P3, P4)
        shaft_neck_intersect = (Pa_optimal + Pb_optimal) / 2 

        femoral_anteversion = femoral_anteversion_angle(center_med, center_lat, shaft_neck_intersect, femur_head, centroid_neck, lat_cond, med_cond)
        # calculate neck length
        neck_length = np.linalg.norm(shaft_neck_intersect)
        # calculate femur length by calculating the distance from FH Center to femoral notch?
        notch_mid = (np.array(femur_mesh.GetPoint(36))   + np.array(femur_mesh.GetPoint(437)) ) / 2
        femur_length = np.linalg.norm(notch_mid)
        if print_all_landmarks == 1:
            print("centroid shaft proximal: ", centroid_shaft_prox)
            print("centroid shaft distal: ", centroid_shaft_dist)
            print("centroid neck: ", centroid_neck)
            print("Lateral Condyle: ", lat_cond)
            print("Medial Condyle: ", med_cond)
            print("Notch Mid: ", notch_mid)
            print("Shaft Neck Intersection: ", shaft_neck_intersect)
        # save all parameters in pd Dataframe
        # Save all parameters in the pd DataFrame
        femur_measurements.loc[femur_list[i], "CCD_Angle"] = neck_shaft_angle
        femur_measurements.loc[femur_list[i], "Anterversion_Angle"] = femoral_anteversion
        femur_measurements.loc[femur_list[i], "Femur_Length"] = femur_length
        femur_measurements.loc[femur_list[i], "Femur_Head_Diameter"] = diameter
        femur_measurements.loc[femur_list[i], "Neck_Length"] = neck_length
    print("done")  
    try:
        femur_measurements.to_excel('00_Data/femur_measurements.xlsx')  
    except:
        femur_measurements.to_excel('femur_measurements.xlsx')  
    

if combine_two_meshes == 1: 
    mesh1_path ="femur_rigidreg_scaled"
    mesh2_path = "hemi_pelvis_rigidreg_scaled"
    mesh1_list = os.listdir(mesh1_path)
    mesh2_list = os.listdir(mesh2_path)
    
    for i in range(len(mesh1_list)):

        mesh1 = o3d.io.read_triangle_mesh(os.path.join(mesh1_path, mesh1_list[i]))

        #read in proximal femur
        mesh2 = o3d.io.read_triangle_mesh(os.path.join(mesh2_path, mesh2_list[i]))
        
        mesh1_name = mesh1_list[i]
        mesh2_name = mesh2_list[i]
        if mesh1_name[:15] != mesh2_name[:15]:
            print("ALARM!")
        # to do 
        combined_mesh = o3d.geometry.TriangleMesh()
        # Append vertices and triangles from the first mesh (femur_mesh)
        combined_mesh.vertices = mesh1.vertices
        combined_mesh.triangles = mesh1.triangles

        # Get the number of vertices in the first mesh for indexing the second mesh
        num_vertices_mesh1 = np.asarray(mesh1.vertices).shape[0]
        #num_vertices_femur = len(num_vertices_femur)

        # Append vertices and triangles from the second mesh (mesh_hemi_pelvis)
        combined_mesh.vertices = o3d.utility.Vector3dVector(
            np.concatenate([np.asarray(combined_mesh.vertices), np.asarray(mesh2.vertices)])
        )
        combined_mesh.triangles = o3d.utility.Vector3iVector(
            np.concatenate([np.asarray(combined_mesh.triangles), np.asarray(mesh2.triangles) + num_vertices_mesh1])
        )
        # Compute normals for the combined mesh
        combined_mesh.compute_vertex_normals()
        output_name = mesh1_name
        output_name = output_name[0:16] + "Hip_Joint" + output_name[19:21] + ".ply"
        # Save the combined mesh to a file
        o3d.io.write_triangle_mesh(os.path.join("full_hip_joint", output_name), combined_mesh)
        # flatten meshes prox and acetabulum
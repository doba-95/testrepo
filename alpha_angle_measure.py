import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d

from matplotlib.pyplot import figure
from scipy.spatial import ConvexHull
import os 
from scipy.optimize import least_squares
from scipy.interpolate import interp2d
from scipy.interpolate import CubicSpline
import ownmodules.yatpkg.util.data as data
from helper_ssm import get_centroid, angle_between_vectors, sphere_fit, get_point_clound
os.chdir(r"F:\Public\14_Femur_Griffith")

import pandas as pd


visual = 1
# min extrusion defines distance to circle considered first exceeding circle -> not sure which value to use yes 
min_extrusion = 1
threshold_contour = 0.7
minus_mask = -8
plus_mask = 7

# find point where distanz between center circle bigger than radius
def find_escape_circle(cx, cy, r, contour):
    center_circle = np.column_stack((cx, cy))
    #contour = np.flipud(head_neck_contour_int)

    for i in range(len(contour)):
        dist = np.linalg.norm(center_circle - contour[i,:])
        if np.linalg.norm(center_circle - contour[i,:]) > (r + min_extrusion):
            return head_neck_contour_int[i,:]
    return head_neck_contour_int[i,:]
        
            


def interpolate_head_neck(x,y):

    spl = CubicSpline(x,y)
    x2 = np.arange(np.min(x), np.max(x), 0.5)

    y_new = spl(x2)

    Z2 = np.column_stack((x2, y_new))
    
    return Z2


list_proximal_femur = os.listdir("05_Transformed_Proximal_Femur")
list_proximal_femur = [femur for femur in list_proximal_femur if "027" in femur or "037" in femur or "055" in femur or "058" in femur ]

femur_alpha_measurements = pd.DataFrame(data = None, index = list_proximal_femur, 
                                      columns= ["3 o'clock", "2 o'clock", "1 o'clock", "12 o'clock" ])

for b, mesh in enumerate(list_proximal_femur):
    # read data file
    mesh_data = o3d.io.read_point_cloud("05_Transformed_Proximal_Femur/" + mesh)
    femur_mesh = data.VTKMeshUtl.load("05_Transformed_Proximal_Femur/" + mesh)
    
    # extract points
    points = np.asarray(mesh_data.points)
    
    # flip y axis 
    points[:,1] *= (-1)
    
    # set initial alpha angle to zero 
    alpha_angle = 0
    # multiplier to adjust rotation direction 
    multiplier = 1
    oclock = 3
    alpha_angle_array = np.ones(4)
    counter = 0
    # insert a for loop to extract alpha angle from 12 o clock until 3 o'clock
    for angle in range(0,91,30):
        
        # rotation matrix that will be applied
        theta = np.radians(multiplier * angle)  # Convert angle to radians
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        points_rot = points @ rotation_matrix

        
        # project points on y axis 
        points_2d = np.column_stack((points_rot[:,0], points_rot[:,2]))

        # calculate midpoint of neck 
        point_ids_femur_neck = np.loadtxt("04_templates/femur_neck.txt")
        # get all coordinates of shaft using the template point id and calculate centroid 
        #get number of Points to iterate over it later
        centroid_shaft_dist= get_centroid(femur_mesh, point_ids_femur_neck) 
        # get center of ball
        point_ids_femur_head = np.loadtxt("04_templates/femur_head.txt")

        p_cloud_femur_head = get_point_clound(femur_mesh, point_ids_femur_head)

        radius, center_femur_head = sphere_fit(p_cloud_femur_head)
        # chatgpt code to rotate the x axis into head shaft vector

        # Step 1: Calculate the vector from centroid_shaft_dist to center_femur_head
        vector = center_femur_head - centroid_shaft_dist

        # Step 2: Normalize the vector
        vector_normalized = vector / np.linalg.norm(vector)

        # Step 3: Calculate the rotation matrix
        # We want to rotate the x-axis [1, 0, 0] to align with the normalized vector
        x_axis = np.array([1, 0, 0])

        # Calculate the cross product and angle between x_axis and vector_normalized
        cross_prod = np.cross(x_axis, vector_normalized)
        dot_prod = np.dot(x_axis, vector_normalized)
        angle = np.arccos(dot_prod)

        # Create the skew-symmetric cross-product matrix
        K = np.array([[0, -cross_prod[2], cross_prod[1]],
                    [cross_prod[2], 0, -cross_prod[0]],
                    [-cross_prod[1], cross_prod[0], 0]])

        # Rotation matrix using the Rodrigues' rotation formula
        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

        # Step 4: Apply the rotation matrix to all points in the mesh
        points_rotated = points_rot @ rotation_matrix.T

        '''here chatgpt insert ends'''
        # Compute the convex hull
        hull = ConvexHull(points_2d)


        # Extract the vertices of the convex hull
        points_2d_contour = points_2d[hull.vertices]

        threshold = ((np.max(points_2d[:,0]) - np.min(points_2d[:,0])) * threshold_contour ) - np.abs(np.min(points_2d[:,0]))
        mask =    points_2d_contour[:,0] > threshold

        points_2d_head = points_2d_contour[mask]

        # Function to calculate the algebraic distance between the 2D points and the circle
        def calc_circle_dist(params, points):
            cx, cy, r = params
            return np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2) - r

        # Initial guess for the circle parameters (center and radius)
        x_m = np.mean(points_2d_contour[:, 0])
        y_m = np.mean(points_2d_contour[:, 1])
        initial_guess = [x_m, y_m, np.mean(np.sqrt((points_2d_contour[:, 0] - x_m)**2 + (points_2d_contour[:, 1] - y_m)**2))]

        #calculate biggest distance between center circle and spehere fit and points_2d_contour
        x_y_sphere = center_femur_head[[0,2]]

        distances_spehere = np.linalg.norm(x_y_sphere - points_2d_contour, axis=1)
        max_distance_spehere = np.argmax(distances_spehere)

        distances_circle = np.linalg.norm(np.array([cx, cy]) - points_2d_contour, axis=1)
        max_distance_circle = np.argmax(distances_circle)

        # Perform least squares optimization to find the circle parameters
        result = least_squares(calc_circle_dist, initial_guess, args=(points_2d_head,))
        cx, cy, r = result.x
        '''newly inserted starts here'''
        #calculate biggest distance between center circle and spehere fit and points_2d_contour
        x_y_sphere = center_femur_head[[0,2]]

        distances_spehere = np.linalg.norm(x_y_sphere - points_2d_contour, axis=1)
        max_distance_spehere = np.argmax(distances_spehere)

        distances_circle = np.linalg.norm(np.array([cx, cy]) - points_2d_contour, axis=1)
        max_distance_circle = np.argmax(distances_circle)
        '''newly inserted stops here'''
        # find points close to head neck junction 
        # creat first mask extracting all x values bigger than 0 
        mask_h_n = (points_2d[:, 0] > minus_mask) & (points_2d[:, 0] < cx + plus_mask ) & (points_2d[:, 1] > cy)
        points_2d_head_neck_contour = points_2d[mask_h_n]
        head_neck_contour = []
        # now find points with highest y value in 0.5 steps in x 

        # Define the range for x-coordinates and step size
        x_min = np.min(points_2d_head_neck_contour[:, 0])
        x_max = np.max(points_2d_head_neck_contour[:, 0]) 
        step = 3
        modulo = x_max // step
        x_min = x_max - (modulo * step) - (3 * step)
        

        for i in np.arange(x_min, x_max, step):
            temp_mask = (points_2d_head_neck_contour[:,0] > i) & (points_2d_head_neck_contour[:,0] < i + step)
            temp_point2d = points_2d_head_neck_contour[temp_mask]
            temp_mask_y = temp_point2d[:,1] == np.max(temp_point2d[:,1])
            temp_point2d = temp_point2d[temp_mask_y]
            head_neck_contour.append(temp_point2d)

        head_neck_contour = np.vstack(head_neck_contour)
        # get smooth contour for calculation
        head_neck_contour_int = np.flipud(interpolate_head_neck(head_neck_contour[:,0], head_neck_contour[:,1]))
        head_neck_contour_int =head_neck_contour_int[:-5]
        #define pooint where neck leaves circle --> here i can try inserting different r/cx, cy
        escape_point = find_escape_circle(cx, cy, r, head_neck_contour_int)
        #define vectors for calculation of alpha angle

        vec_neck_center= np.array([cx, cy]) - np.array([centroid_shaft_dist[0], centroid_shaft_dist[2]])
        vec_center_escape = np.array([cx, cy]) - np.array([escape_point[0], escape_point[1]])
        # not sure why this is still here? probably to only save biggest alpha angle? 
        #if angle_between_vectors(vec_neck_center, vec_center_escape) > alpha_angle:
        alpha_angle = angle_between_vectors(vec_neck_center, vec_center_escape)
        alpha_angle_array[counter] = alpha_angle
        
        if visual == 1:
            plt.figure(figsize=(15,8))
            
            
            # Plot the fitted circle
            circle = plt.Circle((cx, cy), r, color='k', fill=False, lw=3, label='Fitted Circle')
            

            plt.scatter(points_2d[:,0], points_2d[:,1])


            plt.scatter(points_2d_head[:,0], points_2d_head[:,1], c='r')
            plt.scatter(head_neck_contour_int[:,0], head_neck_contour_int[:,1], c='g')

            plt.scatter(cx, cy, c='m')
            plt.scatter(centroid_shaft_dist[0], centroid_shaft_dist[2], c='m')
            plt.scatter(escape_point[0],escape_point[1], marker='*', color='black', s=40)

            plt.gca().add_artist(circle)

            # Plot lines between the points
            plt.plot([centroid_shaft_dist[0], cx], [centroid_shaft_dist[2], cy], 'b-',lw = 3,  label='centroid_shaft_dist to cx, cy')
            plt.plot([cx, escape_point[0]], [cy, escape_point[1]], 'r-', lw = 3, label='cx, cy to escape_point')
            plt.text(cx, cy, round(alpha_angle,2), fontsize=17)
            plt.text(cx, cy+4, mesh[:21], fontsize=17)
            plt.text(cx, cy+8, f"{oclock} o'clock", fontsize=17)

            plt.show()
            oclock = oclock - 1 
        counter = counter +  1
    # save alpha angle in pd data frame    
    femur_alpha_measurements.loc[list_proximal_femur[b]] = alpha_angle_array
#femur_alpha_measurements.to_excel("F:/Public/14_Femur_Griffith/Automated_Alpha_Angle_Resultsthreshold070.xlsx")
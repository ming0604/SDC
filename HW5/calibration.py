import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import click_utils as click



def main():
    #pick the image point
    image = cv2.imread('1518069840621709896.jpg')
    picked_2D_points = click.click_points_2D(image)

    #pick the 3D Lidar point
    points = np.load('1518069840629786841.npy')
    picked_3D_point = click.click_points_3D(points)
    
    #intrinsic matrix of camera
    height, width, channels = image.shape
    fx=fy=698.939
    intrinsic_matrix = np.array([[fx, 0, width/2],
                                 [0, fy, height/2],
                                 [0, 0, 1]])
    dist_coeffs = np.zeros(5)
    
    #solve pnp problem to get transform matrix
    retval,rvec,tvec= cv2.solvePnP(picked_3D_point, picked_2D_points, intrinsic_matrix, dist_coeffs,flags=cv2.SOLVEPNP_EPNP)
    rvec, tvec = cv2.solvePnPRefineLM(picked_3D_point, picked_2D_points, intrinsic_matrix,dist_coeffs,rvec,tvec)

    #use the result of solve pnp problem to build the transform matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    transformation_matrix = np.concatenate((rotation_matrix, tvec),axis=1)
    transformation_matrix = np.concatenate((transformation_matrix, np.array([[0, 0, 0, 1]])), axis=0)
    print('transformation_matrix:')
    print(transformation_matrix)
    np.save('./transformation_matrix.npy', transformation_matrix)



if __name__ == "__main__":
    main()
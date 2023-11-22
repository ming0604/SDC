import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2


def projection(img,pointcloud,ax,index,plot_output_folder):

    transformation_matrix = np.load('transformation_matrix.npy')
    height, width, channels = img.shape
    fx=fy=698.939
    intrinsic_matrix = np.array([[fx, 0, width/2],
                                 [0, fy, height/2],
                                 [0, 0, 1]])
    
    points_world = pointcloud[:,0:3]
    uv = []
    #using the transformation matrix to project the points onto the image
    for i in range(len(points_world)):
        #world coordinates to camera coordinates
        points_camera = transformation_matrix @ (np.append(points_world[i], 1))
        points_camera = points_camera[:-1]

        #camera coordinates to image planes
        image_point = intrinsic_matrix @ (points_camera*(1/points_camera[-1]))
        uv.append(image_point[0:2])
    uv = np.array(uv)

    ax.cla()   #clear the plot
    ax.imshow(img)
    ax.set_xlim(0, 1280)
    ax.set_ylim(720, 0)
    ax.scatter([uv[:, 0]], [uv[:, 1]], c=pointcloud[:, 2], marker=',', s=10, edgecolors='none', alpha=0.7, cmap='jet')
    ax.set_axis_off()

    #save the plot 
    output_file = os.path.join(plot_output_folder, '{:d}.png'.format(index))
    plt.savefig(output_file)

def video(plots_folder, video_folder,fps):
    video_path = video_folder + 'NCTU.mp4'
    output_video = video_path
    
    #get the plots list after sorting
    plots = sorted(os.listdir(plots_folder),key=lambda x: int(x.split('.')[0]))
    
    #get the shape of the plot
    frame = cv2.imread(os.path.join(plots_folder, plots[0]))
    height, width, channels = frame.shape


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for plot in plots:
        path = os.path.join(plots_folder, plot)
        video.write(cv2.imread(path))

    cv2.destroyAllWindows()
    video.release()

def main():
    lidar_folder = './result_NCTU/lidar/'
    image_folder = './result_NCTU/camera/'
    plot_output_folder = './projection_plot/NCTU'
    video_folder = './video/'
    if not os.path.exists(plot_output_folder):
        os.makedirs(plot_output_folder)
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    #sort the image and lidar results by timestamp
    lidar_files = sorted(os.listdir(lidar_folder), key=lambda x: int(x.split('.')[0]))
    image_files = sorted(os.listdir(image_folder), key=lambda x: int(x.split('.')[0]))
    
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot()

    for i,lidar in enumerate(lidar_files):
        # get the timestamp of the lidar file
        lidar_timestamp = int(lidar[:-4])  # remove.npy

        # find the image whose timestamp is closest to the lidar file
        closest_image_file = min(image_files, key=lambda x: abs(int(x[:-4]) - lidar_timestamp))

        # load the lidar and image data
        lidar_data = np.load(os.path.join(lidar_folder, lidar))
        image_data = cv2.imread(os.path.join(image_folder, closest_image_file))
        # because cv2.load is BGR, we need to convert the image data into RGB
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        projection(image_data,lidar_data,ax,i,plot_output_folder)
       
    #transform the projection plots into video
    fps = len(lidar_files)/137
    video(plot_output_folder,video_folder,fps)


if __name__ == "__main__":
    main()
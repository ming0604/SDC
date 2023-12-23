import os
import json
import shutil

def combine_all_training(root_dir,output_dir):
    images_output_folder = os.path.join(output_dir, "images")
    labels_output_folder = os.path.join(output_dir, "labels")
    
    if not os.path.exists(images_output_folder):
        os.makedirs(images_output_folder)
    if not os.path.exists(labels_output_folder):
        os.makedirs(labels_output_folder)

    #copy all training data to one folder
    for folder in os.listdir(root_dir):
        images_folder = os.path.join(root_dir , folder, 'images')
        labels_folder = os.path.join(root_dir, folder, 'labels')

        images = os.listdir(images_folder)
        images.sort()
        labels = os.listdir(labels_folder)
        labels.sort()

        for i in(range(len(images))):
            #copy images
            new_image_file_name = folder + "_" + images[i]
            original_image_path = os.path.join(images_folder, images[i])
            output_image_path = os.path.join(images_output_folder,new_image_file_name)
            shutil.copy(original_image_path, output_image_path)
            
            #copy lebels
            new_label_file_name = folder + "_" + labels[i]
            original_label_path = os.path.join(labels_folder, labels[i])
            output_label_path = os.path.join(labels_output_folder,new_label_file_name)
            shutil.copy(original_label_path, output_label_path)

def main():
    data_training_yolo_folder = os.path.join("./2023_final/data/mini_train_yolo")
    whole_training_output_folder = os.path.join("./2023_final/data/whole_yolo_training")
   
    if not os.path.exists(whole_training_output_folder):
        os.makedirs(whole_training_output_folder)

    combine_all_training(data_training_yolo_folder,whole_training_output_folder)
    
if __name__ == "__main__":
    main()
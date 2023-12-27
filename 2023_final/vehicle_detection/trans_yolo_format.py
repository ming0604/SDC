import os
import json
import shutil

def transform_to_yolo_txt(root_dir,output_dir):
    #folder of dataset
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            radar_folder = os.path.join(root_dir , folder, 'Navtech_Cartesian')
            annotation_path = os.path.join(root_dir, folder, 'annotations', 'annotations.json')
            images_output_folder = os.path.join(output_dir, folder, "images")
            labels_output_folder = os.path.join(output_dir, folder, "labels")

            if not os.path.exists(images_output_folder):
                os.makedirs(images_output_folder)
            if not os.path.exists(labels_output_folder):
                os.makedirs(labels_output_folder)

            with open(annotation_path, 'r') as f_annotation:
                    annotation = json.load(f_annotation)
            radar_images = os.listdir(radar_folder)
            radar_images.sort()

            #classes = ["car", "bus", "van", "truck", "pedestrian", "group_of_pedestrians"]
            #class_to_id = {class_name: index for index, class_name in enumerate(classes)}
            image_width = 1152
            image_height = 1152

            #for each radar image
            for i in range(len(radar_images)):
                # Copy radar image to the output images folder
                radar_image = radar_images[i]
                radar_image_path = os.path.join(radar_folder, radar_image)
                image_output_path = os.path.join(images_output_folder, radar_image)
                shutil.copy(radar_image_path, image_output_path)

                #transform into yolo.txt and then put into corresponding txt.file
                file_name = os.path.splitext(radar_image)[0]
                label_file_path = os.path.join(labels_output_folder, "{:s}.txt".format(file_name))
                
                with open(label_file_path, 'w') as output:
                    for object in annotation:
                        if (object["class_name"] != 'pedestrian' and object["class_name"] != 'group_of_pedestrians'):
                            bbox = object["bboxes"][i]

                            if "position" in bbox:
                                x_upper_left = bbox["position"][0]
                                y_upper_left =  bbox["position"][1]
                                bbox_width = bbox["position"][2]
                                bbox_height = bbox["position"][3]

                                x_c_yolo = (x_upper_left + bbox_width/2)/image_width
                                y_c_yolo = (y_upper_left + bbox_height/2)/image_height
                                width_yolo = bbox_width/image_width
                                height_yolo = bbox_height/image_height
                                yolo_txt = f"{0} {x_c_yolo} {y_c_yolo} {width_yolo} {height_yolo}"

                                output.write(yolo_txt +'\n')
                        
                        

def main():
    data_train_folder = os.path.join("../data/mini_train")
    yolo_train_output_folder = os.path.join("../data/mini_train_yolo")
    if not os.path.exists(yolo_train_output_folder):
        os.makedirs(yolo_train_output_folder)

    data_test_folder = os.path.join("../data/mini_test")
    yolo_test_output_folder = os.path.join("../data/mini_test_yolo")
    if not os.path.exists(yolo_test_output_folder):
        os.makedirs(yolo_test_output_folder)

    transform_to_yolo_txt(data_train_folder,yolo_train_output_folder)   
    transform_to_yolo_txt(data_test_folder,yolo_test_output_folder)
    
if __name__ == "__main__":
    main()
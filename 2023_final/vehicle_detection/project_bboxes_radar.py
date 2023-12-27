import cv2
import os

def draw_bounding_boxes(image_path, label_path):

    radar_image = cv2.imread(image_path)

    with open(label_path, 'r') as labels_file:
        for line in labels_file:
            # get parameter from yolo.txt
            line_split = line.split()
            id = int(line_split[0])
            x_c_yolo = float(line_split[1])
            y_c_yolo = float(line_split[2])
            width_yolo = float(line_split[3])
            height_yolo = float(line_split[4])

            # transform into oringinal radar frame data
            image_width = 1152
            image_height = 1152
            bbox_width = width_yolo*image_width
            bbox_height = height_yolo*image_height
            x_upper_left = int((x_c_yolo * image_width) - bbox_width/2)
            y_upper_left = int((y_c_yolo * image_height) - bbox_height/2)
            x_lower_right = int((x_c_yolo * image_width) + bbox_width/2)
            y_lower_right = int((y_c_yolo * image_height) + bbox_height/2)

            # draw bounding boxes
            color = (0, 0, 255)  #red
            thickness = 2
            image = cv2.rectangle(radar_image, (x_upper_left, y_upper_left), (x_lower_right, y_lower_right), color, thickness)

    # show the result
    cv2.imshow('Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    yolo_images_path = "../data/mini_train_yolo/city_1_3/images"
    yolo_labels_path = "../data/mini_train_yolo/city_1_3/labels" 

    radar_images = os.listdir(yolo_images_path)
    labels = os.listdir(yolo_labels_path)
    for i in range(len(radar_images)):
        image_path = os.path.join(yolo_images_path,radar_images[i])
        label_path = os.path.join(yolo_labels_path,labels[i])
        draw_bounding_boxes(image_path, label_path)

if __name__ == "__main__":
    main()

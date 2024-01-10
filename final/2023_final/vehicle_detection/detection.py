import os
import json
import cv2
from ultralytics import YOLO



def main():
    #pred_image_dir = "../data/Competition_Image"
    #output_json_path = "../data/Competition_prediction/01_02_n_640_rotate.json"

    #pred_image_dir = "../data/mini_test/city_7_0/Navtech_Cartesian"
    #output_json_path = "../data/test_result/yolov8n_640/prediction_city_7_0.json"

    pred_image_dir = "../data/Bonus_Image"
    output_json_path = "../data/Bonus_prediction/Bonus_yolov8_n_1600_ep100.json"
    model = YOLO('./runs/detect/bonus_yolov8n_1600_ep100/weights/best.pt')
 
    #create image list for all images
    image_files = os.listdir(pred_image_dir)
    image_files.sort()
    images_path_list = [os.path.join(pred_image_dir,image) for image in image_files]

    #for visualization
    cv2.namedWindow('YOLO prediction results', cv2.WINDOW_NORMAL)

    output_data = []
    for image in images_path_list:
        #predict the image by yolo
        results = model.predict(image, save=True)
        image_name =  os.path.splitext(os.path.split(image)[1])[0]
        #Note: only one result object in result list
        for result in results:
            boxes = result.boxes.cpu().numpy()
            boxes_xyxy = boxes.xyxy.tolist()
            boxes_conf = boxes.conf.tolist()
            for box, score in zip(boxes_xyxy,boxes_conf):
                min_x = box[0]
                min_y = box[1]
                max_x = box[2]
                max_y = box[3]
                point1 = [min_x,max_y]
                point2 = [min_x,min_y]
                point3 = [max_x,min_y]
                point4 = [max_x,max_y]
                points = [point1,point2,point3,point4]
                objects = {
                            "sample_token": image_name,
                            "points": points,
                            "name": "car",
                            "score": float(score)
                                            }
                output_data.append(objects)

            #plot the results on radar
            result_plot = result.plot()
            cv2.imshow('YOLO prediction results',result_plot)
            if cv2.waitKey(5)==ord('q'):
                cv2.destroyAllWindows()
                break
    cv2.destroyAllWindows()   

    #store the results into .jaso format
    with open(output_json_path, "w") as json_file:
        json.dump(output_data, json_file, indent=2)

if __name__ == "__main__":
    main()
import cv2
import json
import os

def main():
    gt_json_file_path = "../data/mini_test/gt_city_7_0_rot.json"
    prediction_images_path = "../data/test_result/yolov8n_640/predict_city_7_0"
    with open(gt_json_file_path, 'r') as gt:
        gt_data = json.load(gt)
    
    cv2.namedWindow('prediction and gt', cv2.WINDOW_NORMAL)

    images = os.listdir(prediction_images_path)
    for i in range(len(images)):
        image_path = os.path.join(prediction_images_path,images[i])
        image_name = os.path.splitext(images[i])[0]
        image = cv2.imread(image_path)
        for obj in gt_data:
            token = obj.get('sample_token')
            if(token == image_name):
                bbox = obj.get("points")
                min_x = int(min(point[0] for point in bbox))
                min_y = int(min(point[1] for point in bbox))
                max_x = int(max(point[0] for point in bbox))
                max_y = int(max(point[1] for point in bbox))   

                color = (0, 255, 0)  #green
                thickness = 2
                image = cv2.rectangle(image, (min_x, min_y), (max_x, max_y), color, thickness)
        # show the result
        cv2.imshow('prediction and gt', image)
        if cv2.waitKey(50)==ord('q'):
                cv2.destroyAllWindows()
                break
    cv2.destroyAllWindows()            

if __name__ == "__main__":
    main()
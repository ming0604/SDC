# my code running step

因使用YOLOv8，需要先安裝ultralytics庫，並確認 pytorch 和 cuda 也有安裝正確
```
pip install ultralytics
```
## code 使用
### 1. 轉格式至yolo.txt format
**step1.** 先使用==trans_yolo_format.py==將data轉換至yolo format:
    其中輸入檔案路徑及輸出路徑要設好，是整個資料夾的最外面，如下
```python=
data_train_folder = os.path.join("../data/mini_train")
yolo_train_output_folder = os.path.join("../data/mini_train_yolo")
if not os.path.exists(yolo_train_output_folder):
    os.makedirs(yolo_train_output_folder)
```
就能生成對應每個內部資料夾的yolo format 訓練資料夾，如下:
![image](https://hackmd.io/_uploads/Byjy21hOp.png)

接著可用==project_bboxes_radar.py==將轉好的labels投影到image上做確認
也需要設定好對應的images及labels的路徑，如下:
```python=
yolo_images_path = "../data/mini_train_yolo/city_1_3/images"
yolo_labels_path = "../data/mini_train_yolo/city_1_3/labels" 
```
**step2.** 使用==data_combination.py==將所有場景的training data存到一個資料夾:
為了使用所有mini_train的data做training，要將所有data存在一個資料夾下使用，一樣要設好路徑
路徑為剛剛轉完的資料夾當input，並設定output的資料夾路徑，如下:
```python=
data_train_yolo_folder = os.path.join("../data/mini_train_yolo")
whole_train_output_folder = os.path.join("../data/whole_yolo_training")

if not os.path.exists(whole_train_output_folder):
    os.makedirs(whole_train_output_folder)
```
最後得到的所有training data會在設定好的output 資料夾中，如下圖
![image](https://hackmd.io/_uploads/BJen1l3ua.png)

:::warning
bonus因格式不同，要用**bonus_yolo_format.py**去轉格式
:::

### 2. training the model
使用==train.py==
```python=
from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO('yolov8n.yaml')  # build a model from YAML 
    # Train the model
    model.train(data='data.yaml', epochs=300, batch=-1, imgsz=640, device=0)

if __name__ == '__main__':
   main()

```
首先 ==YOLO('yolov8n.yaml')== 中的模型可以自行選擇，如改成yolov8m
須先建立好.yaml檔告訴其training使用的data路徑及class類別數(nc)及類別名稱
我的data.yaml:
```yaml=
train: D:\sdc_final\2023_my_final\data\whole_yolo_training\images
val: D:\sdc_final\2023_my_final\data\whole_yolo_training\images
test: D:\sdc_final\2023_my_final\data\whole_yolo_testing\images

nc: 1
names: ["car"]
```
參數上imgsz及epochs可以自行決定，若使用16系列顯卡要加上==amp=False==這個參數才能跑

### 3. detection
training好後YOLOv8預設會將結果存在`./runs/detect/`下
在我的==detection.py==中model的部分設好weight路徑，以及設定好要做prediction的image資料夾，最後還要設定好輸出的.json檔的路徑，我競賽最好的prediction結果放在
` "./data/Competition_prediction/01_02_n_640_rotate.json"`
```python=
pred_image_dir = "../data/Bonus_Image"
output_json_path = "../data/Bonus_prediction/Bonus_yolov8_n_1600_ep150.json"
model = YOLO('./runs/detect/bonus_yolov8n_1600_ep150/weights/best.pt')
```
而另外30行的參數save=True會把prediction的bbox畫在圖片上並儲存起來
```python=
results = model.predict(image, save=True)
```
預設存在`./runs/detect/predict`

### 4. visualization
這部分另存剛剛用==detection.py==做完後`./runs/detect/predict`的prediction圖片後，再
設定gt.json file的路徑就能跑出視覺化結果了
``` python=
gt_json_file_path = "../data/mini_test/gt_city_7_0_rot.json"
prediction_images_path = "../data/test_result/yolov8n_640/predict_city_7_0"
```

:::success
因檔案data太大，放不上去e3，因此data的部分我就沒放進去zip了，只有保留資料夾名稱，weight也只留了競賽最好結果的那個model
:::

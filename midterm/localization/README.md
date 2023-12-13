# my code running step
主要步驟和助教的相同，先用map_modified.launch生成新地圖，因為有修改地圖不是用助教給的地圖，再用localization.launch定位及visualization可視化
## step1
因為我有修改地圖，需重新生成我的地圖，我的地圖存於pcd_tiles_my_filtered_test/
- 修改map儲存資料夾
    ```
    value="$(find localization)/data/pcd_tiles_my_filtered_test/
    ```
    ```
    roslaunch localization map_modified.launch
    ```
## step2
localization.launch基本上只有修改map路徑資料夾至pcd_tiles_my_filtered_test/
其他就是要跑的bag及result設好就能跑了
     ```
       roslaunch localization localiztion.launch
    ```
## step3
visualization.launch基本上也一樣修改map路徑資料夾至pcd_tiles_my_filtered_test/
其他就是要跑的bag及result設好就能跑了
     ```
       roslaunch localization visualization.launch
    ```
    
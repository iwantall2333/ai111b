
聲明 : yolov7的檔案取自[官網](https://github.com/WongKinYiu/yolov7)，因為我最終更動到的檔案只有`detect.py`，所以我在github上只放上`detect.py`(`my_detect.py`)

# 主題 : 使用yolov7官方預訓練模型進行人的偵測


### 步驟

1. 在anaconda 安裝yolo v7, 參考 : [安裝YOLO v7](https://blog.eddie.tw/yolo-v7-install/#more-14)@Eddie's Blog

2. 接著為了要使用GPU運行，要安裝CUDA+cudnn，pytorch
    - 安裝CUDA+cudnn : (CUDA 11.8、10.2為推薦版本)

    - 如果pytorch、torchvision都一直顯示cpu版本，就把卸載的本地torch、torchvision，再重新安裝在Anaconda中，(否則裝在本的的版本會蓋住裝在其他環境的版本)

3. 修改detect.py的內容

    ```py
    import argparse
    import time
    from pathlib import Path

    import cv2
    import torch
    import torch.backends.cudnn as cudnn
    from numpy import random

    torch.cuda.is_available() #需要添加這行
    ```

4. 在設好的環境下打`python my_detect.py --weights yolov7.pt --conf 0.5 --source 0 --view-img --nosave --device 1` :

    - `device 1` 是nvidia gpu(工作管理員可以看到標號)
    - `--weights yolov7.pt`指權重使用`yolov7.pt`
    - `--source 0` 代表使用webcam偵測
    - `--nosave` 則是讓每次偵測都不會儲存結果，因為我們是要用webcam偵測，如果儲存結果電腦硬碟容量會越來越少

運行成功後會發現，一般筆電跑起來還是很慢的，所以我們要改用更小的權重檔

5. 在設好的環境下打`python my_detect.py --weights yolov7-tiny.pt --conf 0.5 --source 0 --view-img --nosave --device 1` :

    - `yolov7-tiny.pt` 是比  `yolov7.pt` 更加輕便的權重檔，所以跑起來會快超多

6. 修改detect.py的內容

    ```py
    parser.add_argument('--classes', nargs='+', type=int,default='0', help='filter by class: --class 0, or --class 0 2 3')
    ```
    yolov7預設的分類載檢測目標有80類，`default='0'`代表我只偵測第一個類別(也就是人)
    修改好後，就可以指定只偵測人了

7. demo影片 : [yolov7-tiny](https://drive.google.com/file/d/1P6DZCHwPJep_4WP5qqSs9tks6-JPUCXA/view)

### 結論

yolo模型在即時影像偵測很有優勢，再加上前面做的實驗與demo片的示範，可以得出一個結論，一般家庭其實用官方預設模型，不用自己訓練資料，只需對程式做一點修改，就能獲得一個很不錯的，會物件偵測的監視器了

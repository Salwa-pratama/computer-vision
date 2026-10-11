Hallo hari ini saya akan belajar mengenai computer vision menggunakan mobilenet-SSD, dan media pipe



Dibawah ini merupakan model MobileNetSSD yang saya pakai
``` bash
# MobileNetSSD_deploy.prototxt
wget https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt

# MobileNetSSD_deploy.caffemodel
wget -O MobileNetSSD_deploy.caffemodel https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.caffemodel

```

Dan dibawah ini merupakan model COCO-SSD yang saya pakai
``` bash
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
# extraksi jika menggunakan containerisasi
tar --no-same-owner -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt

```

# AI-Video-Analytics 
### Covid-19 Spread Tracker
Covid-19 Spread Tracker is the name of our project which is basically
is a computer vision project.
Covid-19 Spread Tracker is mainly used to detect the people who are violating the safe distance between each other,
Also, it detects the people who are wearing masks and people who are not wearing masks.

####**Models used for Covid-19 Spread Tracker software:**
- YOLOv5 is used to detect the persons and the locations of them.
- DBSCAN is used to detect unsafe people and clustering (the unsafe distance between persons is 6 feet). The threshold of the DBSCAN algorithm is 150 which detects the 6 feet in the real world.
- RetinaNetMobileNetV1 model is used to detect the faces of the persons.
-ResNet50 model was trained to classify the masked and the unmasked persons.

#### Dataset:
10000 images of masked and unmasked faces are used to train the last layer of our ResNet50 model.
Blurring was applied on the dataset that is used to train ResNet50 so it can adapt the low quality of CCTV cameras.

Covid-19 Spread Tracker software detects:
Total number of people
number of safe people
number of unsafe people
number of masked people
number of unmasked people


###Team:

Shrouk El Masry

Samar Ibrahim

mohamed El-Ghannam

Sara Hadou

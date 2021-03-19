# Project of Vision and Cognitive System

## Setup
  Install virtualenv
```bash
  $ python3 -m pip install --upgrade pip
  $ pip3 install virtualenv
```

  Setup project
``` bash
  $ git clone https://github.com/ZippoCode/project_vcs.git
  $ virtualenv -p python3.6 project_vcs
  $ cd project_vcs
  $ source bin/activate
  $ pip install requirements.txt 
```

You need install a FFMPEG codec. If you use Linux S.O. you can use ``` sudo apt-get install FFMPEG ```.
Else if you are using Windows OS you need download FFMPEG codec from this [link](https://ottverse.com/ffmpeg-builds/)
``` 
  Download the Build File
  Extract the file
  Rename the folder as 'ffmpeg'
  Move the folder in 'project_vcs'
```

## Structure
* Input: all dataset was put in the `data` folder (video, image, map, csv)
* Output: in `output` folder
* Code file: all code files are in `src` folder
* Yolo folder contains weight, coco.name and config files


## Painting Detection & Rectification
The file `src/test_painting_detection.py` contains all functions for detecting edge and finding four corners of the painting.
In addition, file `test_painting_rectification.py` has affine transformations to create a new image that has only painting. <br>Usage:<br/>
```bash
  $ python test_painting_detection.py
  $ python test_painting_rectification.py
```
  Output video and PCK file for the Painting Detection is : `../output/painting_detected/`.
  Output video for the Painting Rectification is `./output/painting_rectificated/`.
  
![Figure 1](https://github.com/ZippoCode/project_vcs/blob/master/image_results/original_frame_1.png)
![Figure 2](https://github.com/ZippoCode/project_vcs/blob/master/image_results/painting_detection.png)

## Retrieval Painting
The task's goal is given a query image find the same similar images in the set. The first step is detected the features of query image and these of image contains in the database and then matches these. 
For the extract the features we used a **Scale-Invariant Feature Transform** [SIFT](https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html)
<br/>Usage
```bash
  $ python test_painting_retrieval.py
```
**Parameters:**:  
`--num`: The number of paintings<br/>
`--painting`: The path of painting<br/>
`--folder`: The path of folder which contains paintings<br/>
`--source_db`: The path of paintings database folder<br/>
<br/>
Output is an image that contains image in result of painting rectification and painting in dataset<br/>
![Figure 5](https://github.com/ZippoCode/project_vcs/blob/master/image_results/painting_retrieval_1.png)
![Figure 6](https://github.com/ZippoCode/project_vcs/blob/master/image_results/painting_retrieval_2.png)

<br>Results:<br/>

| Accuracy  |  Recall   | Precision | F1        |
| --------- | --------- | --------- | --------- |
| 0.601     | 0.828     | 0.568     | 1.0       |

## People detection
We use YOLOv3 to detect people.
  * Weight file: `../yolo/yolov3-obj-train_final.weights`. You can download it [here](https://drive.google.com/file/d/1bkADs1lT8ayXwDwYnS1rU0Nz8rmTNmOt/view?usp=sharing)
  * Coco file: `../yolo/coco.name`
  * Config file: `../yolo/cfg/yolov3-obj-test.cfg`

Run
```bash
  $ python test_people_detection.py
```
  Output: video result in `../output/person_detected/` and we create file PCK that contains coordinate of people in video in `../output/person_detected/{video_name}.pck` file .
![Figure 6](https://github.com/ZippoCode/project_vcs/blob/master/image_results/yolov3.png)

## People localization
Make sure you have file PCK in `../output/person_detected/` with the same video name that you will use to locate.

![Figure 7](https://github.com/ZippoCode/project_vcs/blob/master/image_results/painting_location.png)

Run
```bash
  $ python test_people_localization.py 
```
Output is a map with a green highlight rectangle which is the room that real person and the list of painting with best similarity value.

![Figure 8](https://github.com/ZippoCode/project_vcs/blob/master/image_results/map.png)
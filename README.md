# Project of Vision and Cognitive System

## Edge Detection

Steps:
 
1. **Mean Shift Filtering**
2. **Threshould**: Apply the threshould in HSV colorspace




## Retrieval Painting
The task's goal is given a query image find the same similar images in the set. This is called Image Retrieval. The first step is detected the features of query image and these of image contains in the database and then matches these. 
For the extract the features we used a **Scale-Invariant Feature Transform** ([SIFT](https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html))  while for the Features Match we used **FLANN** which filter the matches second the propos of Lowe.



### References:
* [Mean Shift Analysis and Application](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.34.1965&rep=rep1&type=pdf)
* [Mean Shift: A Robust Approach toward Feature Space Analysis](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.76.8968&rep=rep1&type=pdf)
* [A Novel Approach for Color Image Edge Detection Using Multidirectional Sobel Filter on HSV Color Space](https://www.researchgate.net/profile/Dibya_Bora4/publication/314446743_A_Novel_Approach_for_Color_Image_Edge_Detection_Using_Multidirectional_Sobel_Filter_on_HSV_Color_Space/links/58c26c7245851538eb7cf0bd/A-Novel-Approach-for-Color-Image-Edge-Detection-Using-Multidirectional-Sobel-Filter-on-HSV-Color-Space.pdf)

### OpenCV Code
* [pyrMeanShiftFiltering](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=meanshiftfiltering#pyrmeanshiftfiltering)
* [Threshoulding Operations using inRange](https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html)
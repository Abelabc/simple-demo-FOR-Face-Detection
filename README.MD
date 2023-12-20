## simple-demo-FOR-Face-Detection

### *Flowchart*

![flowchart](https://github.com/Abelabc/simple-demo-FOR-Face-Detection/blob/main/pic/flowchart.png)

### *Calling the camera or video*

```
    #从视频读取
    #cap = cv2.VideoCapture("VCG42N1270675644.mp4")
    # 从电脑默认0号摄像头读取
    cap = cv2.VideoCapture(0)
```

![video](https://github.com/Abelabc/simple-demo-FOR-Face-Detection/blob/main/pic/video.png)

### *Using* *haar* *classfier*

Haar classifier= Haar-like features \+ AdaBoost algorithm \+ cascade  \+ integral graph fast computation

[haar](https://zhuanlan.zhihu.com/p/51431663)[分类器原理和](https://zhuanlan.zhihu.com/p/51431663)[opencv](https://zhuanlan.zhihu.com/p/51431663)[实现 ](https://zhuanlan.zhihu.com/p/51431663)[- ](https://zhuanlan.zhihu.com/p/51431663)[知乎 ](https://zhuanlan.zhihu.com/p/51431663)[(zhihu.com)](https://zhuanlan.zhihu.com/p/51431663)

### *Prepapre* *for face detection*
Grayscale is the process of making the R, G, and B components of the color image equal, so that R = G = B.At this time, the color represents the grayscale color.

```
grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
```

### *Detection and drawing*

detectMultiScale function(face detection used in opencv2) 

It detects all the faces in the picture and stores the faces in a vector with the coordinates and size of each face (represented by a rectangle), and the function is called by the classifier object 


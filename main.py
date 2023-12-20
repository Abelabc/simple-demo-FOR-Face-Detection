import cv2


def face_detection():
    # 1.调用摄像头或者使用读取视频
    cv2.namedWindow("CaptureFace")  # 给窗口命名
    #cap = cv2.VideoCapture("VCG42N1270675644.mp4")  # 从电脑默认0号摄像头读取
    cap = cv2.VideoCapture(0)

    # 2.人脸识别器分类器q
    classfier = cv2.CascadeClassifier(
      'venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')  # haar分类器
    color = (0, 255, 0)

    while cap.isOpened():  # 判断当前的摄像头是否初始化成功
        flag, frame = cap.read()   # 参数flag 为True 或者False,代表有没有读取到图片，第二个参数frame表示截取到一帧的图片

        frame = cv2.flip(frame,1)  # 镜像操作1为水平反转

        if not flag:
            break

        # 3.灰度转换
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片

        # 4.人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        facerects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(facerects) > 0:  # 大于0则检测到人脸
            for faceRect in facerects:  # 单独框出每一张人脸
                x, y, w, h = faceRect
        # 5.画图
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3)
        cv2.imshow("CaptureFace", frame)   # 显示这一帧的画面
        if cv2.waitKey(10) & 0xFF == ord('q'):  # waitkey控制这一帧画面显示时间、按下q键后break
            break
    cap.release()


face_detection()  # 使用摄像头


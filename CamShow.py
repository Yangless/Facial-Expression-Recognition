from OboardCamDisp import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from PyQt5.QtCore import QTimer,QCoreApplication
from PyQt5.QtGui import QPixmap
import cv2
import qimage2ndarray
import time
import argparse
from predict import vgg2pre,res2pre,se_resnet2pre,densenet2pre,predict




#open_dnn
model_bin = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_text = "deploy.prototxt"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./', help='test video path.')
    parser.add_argument('--video', type=str, default=None,
                        help='test video path')
    return parser.parse_args()


args = parse_args()




# load caffe model
net2 = cv2.dnn.readNetFromCaffe(config_text, model_bin)

# set back-end
net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


#face_cascade = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('../opencv-master/data/haarcascades/haarcascade_eye.xml')
vgg=vgg2pre()
resnet=res2pre()
se_resnet=se_resnet2pre()
densenet=densenet2pre()
#resnet=vgg2pre()
class CamShow(QMainWindow,Ui_MainWindow):
    def __del__(self):
        try:
            self.camera.release()  # 释放资源
        except:
            return
    def __init__(self,parent=None):
        super(CamShow,self).__init__(parent)
        self.setupUi(self)
        self.PrepSliders()
        self.PrepWidgets()
        self.PrepParameters()
        self.CallBackFunctions()
        self.Timer=QTimer()
        self.Timer.timeout.connect(self.TimerOutFun)
        self.Face=[]
    def PrepSliders(self):
        self.RedColorSld.valueChanged.connect(self.RedColorSpB.setValue)
        self.RedColorSpB.valueChanged.connect(self.RedColorSld.setValue)
        self.GreenColorSld.valueChanged.connect(self.GreenColorSpB.setValue)
        self.GreenColorSpB.valueChanged.connect(self.GreenColorSld.setValue)
        self.BlueColorSld.valueChanged.connect(self.BlueColorSpB.setValue)
        self.BlueColorSpB.valueChanged.connect(self.BlueColorSld.setValue)
        self.ExpTimeSld.valueChanged.connect(self.ExpTimeSpB.setValue)
        self.ExpTimeSpB.valueChanged.connect(self.ExpTimeSld.setValue)
        self.GainSld.valueChanged.connect(self.GainSpB.setValue)
        self.GainSpB.valueChanged.connect(self.GainSld.setValue)
        self.BrightSld.valueChanged.connect(self.BrightSpB.setValue)
        self.BrightSpB.valueChanged.connect(self.BrightSld.setValue)
        self.ContrastSld.valueChanged.connect(self.ContrastSpB.setValue)
        self.ContrastSpB.valueChanged.connect(self.ContrastSld.setValue)
    def PrepWidgets(self):
        self.PrepCamera()
        self.StopBt.setEnabled(False)
        self.RecordBt.setEnabled(False)
        self.GrayImgCkB.setEnabled(False)
        self.RedColorSld.setEnabled(False)
        self.RedColorSpB.setEnabled(False)
        self.GreenColorSld.setEnabled(False)
        self.GreenColorSpB.setEnabled(False)
        self.BlueColorSld.setEnabled(False)
        self.BlueColorSpB.setEnabled(False)
        self.ExpTimeSld.setEnabled(False)
        self.ExpTimeSpB.setEnabled(False)
        self.GainSld.setEnabled(False)
        self.GainSpB.setEnabled(False)
        self.BrightSld.setEnabled(False)
        self.BrightSpB.setEnabled(False)
        self.ContrastSld.setEnabled(False)
        self.ContrastSpB.setEnabled(False)
    def PrepCamera(self):
        try:
            global args
            if args.video :
                self.camera=cv2.VideoCapture(args.path+args.video)
            else:
                self.camera=cv2.VideoCapture(0)
                
            self.MsgTE.clear()
            self.MsgTE.append('Oboard camera connected.')
            self.MsgTE.setPlainText()
        except Exception as e:
            self.MsgTE.clear()
            self.MsgTE.append(str(e))
    def PrepParameters(self):
        self.RecordFlag=0
        self.RecordPath=args.video
        self.FilePathLE.setText(self.RecordPath)
        self.Image_num=0
        self.R=1
        self.G=1
        self.B=1

        self.ExpTimeSld.setValue(self.camera.get(15))
        self.SetExposure()
        self.GainSld.setValue(self.camera.get(14))
        self.SetGain()
        self.BrightSld.setValue(self.camera.get(10))
        self.SetBrightness()
        self.ContrastSld.setValue(self.camera.get(11))
        self.SetContrast()
        self.MsgTE.clear()
    def CallBackFunctions(self):
        self.FilePathBt.clicked.connect(self.SetFilePath)
        self.ShowBt.clicked.connect(self.StartCamera)
        self.StopBt.clicked.connect(self.StopCamera)
        self.RecordBt.clicked.connect(self.RecordCamera)
        self.ExitBt.clicked.connect(self.ExitApp)
        self.GrayImgCkB.stateChanged.connect(self.SetGray)
        self.ExpTimeSld.valueChanged.connect(self.SetExposure)
        self.GainSld.valueChanged.connect(self.SetGain)
        self.BrightSld.valueChanged.connect(self.SetBrightness)
        self.ContrastSld.valueChanged.connect(self.SetContrast)
        self.RedColorSld.valueChanged.connect(self.SetR)
        self.GreenColorSld.valueChanged.connect(self.SetG)
        self.BlueColorSld.valueChanged.connect(self.SetB)
    def SetR(self):
        R=self.RedColorSld.value()
        self.R=R/255
    def SetG(self):
        G=self.GreenColorSld.value()
        self.G=G/255
    def SetB(self):
        B=self.BlueColorSld.value()
        self.B=B/255
    def SetContrast(self):
        contrast_toset=self.ContrastSld.value()
        try:
            self.camera.set(11,contrast_toset)
            self.MsgTE.setPlainText('The contrast is set to ' + str(self.camera.get(11)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def SetBrightness(self):
        brightness_toset=self.BrightSld.value()
        try:
            self.camera.set(10,brightness_toset)
            self.MsgTE.setPlainText('The brightness is set to ' + str(self.camera.get(10)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def SetGain(self):
        gain_toset=self.GainSld.value()
        try:
            self.camera.set(14,gain_toset)
            self.MsgTE.setPlainText('The gain is set to '+str(self.camera.get(14)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def SetExposure(self):
        try:
            exposure_time_toset=self.ExpTimeSld.value()
            self.camera.set(15,exposure_time_toset)
            self.MsgTE.setPlainText('The exposure time is set to '+str(self.camera.get(15)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def SetGray(self):
        if self.GrayImgCkB.isChecked():
            self.RedColorSld.setEnabled(False)
            self.RedColorSpB.setEnabled(False)
            self.GreenColorSld.setEnabled(False)
            self.GreenColorSpB.setEnabled(False)
            self.BlueColorSld.setEnabled(False)
            self.BlueColorSpB.setEnabled(False)
        else:
            self.RedColorSld.setEnabled(True)
            self.RedColorSpB.setEnabled(True)
            self.GreenColorSld.setEnabled(True)
            self.GreenColorSpB.setEnabled(True)
            self.BlueColorSld.setEnabled(True)
            self.BlueColorSpB.setEnabled(True)
    def StartCamera(self):
        self.ShowBt.setEnabled(False)
        self.StopBt.setEnabled(True)                          #########
        self.RecordBt.setEnabled(True)
        self.GrayImgCkB.setEnabled(True)
        if self.GrayImgCkB.isChecked()==0:
            self.RedColorSld.setEnabled(True)
            self.RedColorSpB.setEnabled(True)
            self.GreenColorSld.setEnabled(True)
            self.GreenColorSpB.setEnabled(True)
            self.BlueColorSld.setEnabled(True)
            self.BlueColorSpB.setEnabled(True)
        self.ExpTimeSld.setEnabled(True)
        self.ExpTimeSpB.setEnabled(True)
        self.GainSld.setEnabled(True)
        self.GainSpB.setEnabled(True)
        self.BrightSld.setEnabled(True)
        self.BrightSpB.setEnabled(True)
        self.ContrastSld.setEnabled(True)
        self.ContrastSpB.setEnabled(True)
        self.RecordBt.setText('VGG19')

        self.Timer.start(50)
        self.timelb=time.clock()
 #       self.timelb=time.perf_counter()

    def SetFilePath(self):
        dirname = QFileDialog.getExistingDirectory(self, "浏览", '.')
        if dirname:
            self.FilePathLE.setText(dirname)
            self.RecordPath=dirname+'/'
    def TimerOutFun(self):
        success,img=self.camera.read()
        if success:
            self.Image = self.ColorAdjust(img)
            #self.Image=img
            self.DispImg()
            self.Image_num+=1
            if self.RecordFlag:
                self.video_writer.write(img)
            if self.Image_num%10==9:
                frame_rate=10/(time.clock()-self.timelb)
                self.FmRateLCD.display(frame_rate)
                self.timelb=time.clock()
 #               self.timelb=time.perf_counter()
                #size=img.shape
                self.ImgWidthLCD.display(self.camera.get(3))
                self.ImgHeightLCD.display(self.camera.get(4))
        else:
            self.MsgTE.clear()
            self.MsgTE.setPlainText('Image obtaining failed.')
    def ColorAdjust(self,img):
        try:
            B=img[:,:,0]
            G=img[:,:,1]
            R=img[:,:,2]
            B=B*self.B
            G=G*self.G
            R=R*self.R
            #B.astype(cv2.PARAM_UNSIGNED_INT)
            #G.astype(cv2.PARAM_UNSIGNED_INT)
            #R.astype(cv2.PARAM_UNSIGNED_INT)

            img1=img
            img1[:,:,0]=B
            img1[:,:,1]=G
            img1[:,:,2]=R
            return img1
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def DispImg(self):
        net_type=self.RecordBt.text()
        h, w = self.Image.shape[:2]
        blobImage = cv2.dnn.blobFromImage(self.Image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False);
        net2.setInput(blobImage)
        self.Face  = net2.forward()
        
        for detection in self.Face[0,0,:,:]:
            score = float(detection[2])
           
#            objIndex = int(detection[1])
            if score > 0.5:
                left = detection[3]*w
                top = detection[4]*h
                right = detection[5]*w
                bottom = detection[6]*h
                cv2.rectangle(self.Image, (int(left)-12, int(top)-12), (int(right)+12, int(bottom)+12), (255, 0, 0), thickness=2)
                cv2.putText(self.Image, "%.2f"%score, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                if self.StopBt.text()=='继续':
                    roi = self.Image[int(top)-10:int(bottom)+10,int(left)-10:int(right)+10]
                    
                    if roi.shape[0]<44 or roi.shape[1]<44:
                        print(roi.shape)
                        continue
                  #  cv2.imwrite(str(i)+".jpg", roi)
                    net_type=self.RecordBt.text()

                    if(net_type=="VGG19"):
                        pred=predict(roi,vgg.net)
                    elif(net_type=="DenseNet121"):
                        pred=predict(roi,densenet.net)
                    elif(net_type=="ResNet18"):
                        pred=predict(roi,resnet.net)
                    elif(net_type=="SeNet18"):
                        pred=predict(roi,se_resnet.net)
            
                    cv2.putText(self.Image, '%s'%pred, (int(left), int(top)-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (55,255,155), 2)

#                self.MsgTE.clear()
                    self.MsgTE.setPlainText(pred)


        if self.GrayImgCkB.isChecked():
            img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        qimg = qimage2ndarray.array2qimage(img)
        self.DispLb.setPixmap(QPixmap(qimg))
        self.DispLb.show()
    def StopCamera(self):
        if self.StopBt.text()=='预测':
            self.StopBt.setText('继续')

        elif self.StopBt.text()=='继续':
            self.StopBt.setText('预测')
#            self.RecordBt.setText('录像')
            self.Timer.start(1)
    def RecordCamera(self):
        tag=self.RecordBt.text()
        if tag=='VGG19':
            self.RecordBt.setText('ResNet18')
            self.StopBt.setText('预测')
        elif tag=='ResNet18':
            self.RecordBt.setText('DenseNet121')
#            self.RecordBt.setText('录像')
            self.StopBt.setText('预测')
        elif tag=='DenseNet121':
            self.RecordBt.setText('SeNet18')
#            self.RecordBt.setText('录像')
            self.StopBt.setText('预测')
        elif tag=='SeNet18':
            self.RecordBt.setText('VGG19')
#            self.RecordBt.setText('录像')
            self.StopBt.setText('预测')


    def ExitApp(self):
        self.Timer.stop()
        self.camera.release()
        self.MsgTE.setPlainText('Exiting the application..')
        QCoreApplication.quit()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    ui=CamShow()
    ui.show()
#    sys.exit(app.exec_())
    app.exec_()
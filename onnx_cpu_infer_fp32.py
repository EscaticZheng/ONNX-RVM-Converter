import onnxruntime as ort
import numpy as np
import cv2
from tqdm import tqdm
#load model
sess = ort.InferenceSession('rvm_mobilenetv3_fp32.onnx',providers=['CPUExecutionProvider'])

#rec:rnn input，downsample_ratio:1280x720=0.375,1920x1080=0.25,4k=0.125
rec = [ np.zeros([1, 1, 1, 1], dtype=np.float32) ] * 4  # Must match dtype of the model.
downsample_ratio = np.array([0.375], dtype=np.float32)  # dtype always FP32

cap = cv2.VideoCapture('input.avi',cv2.CAP_FFMPEG)
out = cv2.VideoWriter('com.avi',cv2.VideoWriter_fourcc(*'XVID'),30,(1280,720),True)   #using ffmpeg to read
bgr = np.array([.47,.1,.6]) #bgr:background
bgr = np.resize(bgr,[3,1,1]) #resize shape
#progress bar
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total = frame_count)
i = 1
#inference loop
while(True):
    pbar.update(i)
    ret, src = cap.read() #ret:True or False，src:frame
    if ret == False:
        break; 
    src = src.swapaxes(1,2).swapaxes(0,1) #swap dimension
    src = src.astype('float32')/255 #0-255 to 0 -1
    src = src.reshape([1,3,720,1280]) #BCHW shape
    fgr, pha, *rec = sess.run([], {
        'src': src, 
        'r1i': rec[0], 
        'r2i': rec[1], 
        'r3i': rec[2], 
        'r4i': rec[3], 
        'downsample_ratio': downsample_ratio
    })
    #compute fgr and pha to get outcome frame
    com = (fgr * pha + bgr * (1 - pha))*255
    com = com.reshape([3,720,1280]) #repeat converting process
    com = com.astype('int8')
    com = com.swapaxes(0,1).swapaxes(1,2)
    out.write(com)
cap.release()
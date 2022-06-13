import onnxruntime as ort
import numpy as np
import cupy as cp
import cv2
import time
#load model
sess = ort.InferenceSession('rvm_mobilenetv3_fp16.onnx',providers=['CUDAExecutionProvider'])

#bing io
io = sess.io_binding()

#rec:rnn input，downsample_ratio:1280x720=0.375,1920x1080=0.25,4k=0.125
rec = [ ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 1, 1, 1], dtype=np.float16), 'cuda') ] * 4
downsample_ratio = ort.OrtValue.ortvalue_from_numpy(np.asarray([0.375], dtype=np.float32), 'cuda')

#bind output.
for name in ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']:
    io.bind_output(name, 'cuda')

cap = cv2.VideoCapture('input.mp4',cv2.CAP_FFMPEG)   #using ffmpeg to read
out = cv2.VideoWriter('com.avi',cv2.VideoWriter_fourcc(*'XVID'),30,(1280,720),True)
bgr = cp.array([.47,.1,.6]) #bgr:background
bgr = cp.resize(bgr,[3,1,1]) #resize shape
#inference loop
while(True):
    ret, src = cap.read() #ret:True or False，src:frame
    if ret == False:
        break; 
    src = cp.array(src) #convert numpy to cupy
    src = src.swapaxes(1,2).swapaxes(0,1) #swap dimension
    src = src.astype('float16')/255 #0-255 to 0 -1
    src = src.reshape([1,3,720,1280]) #BCHW shape
    src = cp.asnumpy(src)  #将convert cupy to numy
    io.bind_cpu_input('src', src)
    io.bind_ortvalue_input('r1i', rec[0])
    io.bind_ortvalue_input('r2i', rec[1])
    io.bind_ortvalue_input('r3i', rec[2])
    io.bind_ortvalue_input('r4i', rec[3])
    io.bind_ortvalue_input('downsample_ratio', downsample_ratio)

    sess.run_with_iobinding(io)

    fgr, pha, *rec = io.get_outputs()
    #compute fgr and pha to get outcome frame
    fgr = fgr.numpy()
    pha = pha.numpy()
    fgr = cp.array(fgr)
    pha = cp.array(pha)
    com = (fgr * pha + bgr * (1 - pha))*255
    com = com.reshape([3,720,1280]) #repeat converting process
    com = com.astype('int8')
    com = com.swapaxes(0,1).swapaxes(1,2)
    com = cp.asnumpy(com)
    out.write(com)
cap.release()
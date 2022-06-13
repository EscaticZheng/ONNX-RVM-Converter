import onnxruntime as ort
import numpy as np
import cupy as cp
import cv2
import time
sess = ort.InferenceSession('rvm_mobilenetv3_fp32.onnx',providers=['CUDAExecutionProvider'])

io = sess.io_binding()

rec = [ ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 1, 1, 1], dtype=np.float32), 'cuda') ] * 4
downsample_ratio = ort.OrtValue.ortvalue_from_numpy(np.asarray([0.375], dtype=np.float32), 'cuda')


for name in ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']:
    io.bind_output(name, 'cuda')


cap = cv2.VideoCapture('input.mp4',cv2.CAP_FFMPEG) 
out = cv2.VideoWriter('com.avi',cv2.VideoWriter_fourcc(*'XVID'),30,(1280,720),True)
while(True):
    ret, src = cap.read()
    if ret == False:
        break;
    src = cp.array(src)
    src = src.swapaxes(1,2).swapaxes(0,1)
    src = src.astype('float32')/255
    src = src.reshape([1,3,720,1280])
    src = cp.asnumpy(src)
    io.bind_cpu_input('src', src)
    io.bind_ortvalue_input('r1i', rec[0])
    io.bind_ortvalue_input('r2i', rec[1])
    io.bind_ortvalue_input('r3i', rec[2])
    io.bind_ortvalue_input('r4i', rec[3])
    io.bind_ortvalue_input('downsample_ratio', downsample_ratio)

    sess.run_with_iobinding(io)

    fgr, pha, *rec = io.get_outputs()
    fgr = fgr.numpy()
    pha = pha.numpy()
    fgr = cp.array(fgr)
    pha = cp.array(pha)
    bgr = cp.array([.47,.1,.6])
    bgr = cp.resize(bgr,[3,1,1])
    com = fgr * pha + bgr * (1 - pha)
    com = com.reshape([3,720,1280])
    com = com*255
    com = com.astype('int8')
    com = com.swapaxes(0,1).swapaxes(1,2)
    com = cp.asnumpy(com)
    out.write(com)
cap.release()
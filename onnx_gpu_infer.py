import onnxruntime as ort
import numpy as np
import cupy as cp
import cv2
import time
#加载模型
sess = ort.InferenceSession('rvm_mobilenetv3_fp16.onnx',providers=['CUDAExecutionProvider'])

#创建io
io = sess.io_binding()

#rec为模型循环记忆输入，downsample_ratio为下采样比,1280x720=0.375,1920x1080=0.25,4k=0.125
rec = [ ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 1, 1, 1], dtype=np.float16), 'cuda') ] * 4
downsample_ratio = ort.OrtValue.ortvalue_from_numpy(np.asarray([0.375], dtype=np.float32), 'cuda')

#绑定模型输出.
for name in ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']:
    io.bind_output(name, 'cuda')

#推断循环
cap = cv2.VideoCapture('input.mp4',cv2.CAP_FFMPEG)   #调用本地文件，使用ffmpeg加速输入
out = cv2.VideoWriter('com.avi',cv2.VideoWriter_fourcc(*'XVID'),30,(1280,720),True)
bgr = cp.array([.47,.1,.6]) #bgr为背景
bgr = cp.resize(bgr,[3,1,1]) #resize维度
while(True):
    ret, src = cap.read() #ret为True or False，src为读取的每帧图像的数组
    if ret == False:
        break; #帧读取完的时候ret就会从True变成False
    src = cp.array(src) #使用cupy加速
    src = src.swapaxes(1,2).swapaxes(0,1) #交换维度顺序
    src = src.astype('float16')/255 #归一化并转换为FP16
    src = src.reshape([1,3,720,1280]) #转化为BCHW输入
    src = cp.asnumpy(src)  #将cupy重新转化为numpy(模型只接受numpy输入)
    io.bind_cpu_input('src', src)
    io.bind_ortvalue_input('r1i', rec[0])
    io.bind_ortvalue_input('r2i', rec[1])
    io.bind_ortvalue_input('r3i', rec[2])
    io.bind_ortvalue_input('r4i', rec[3])
    io.bind_ortvalue_input('downsample_ratio', downsample_ratio)

    sess.run_with_iobinding(io)

    fgr, pha, *rec = io.get_outputs()
    #fgr为前景，pha为透明度
    fgr = fgr.numpy()
    pha = pha.numpy()
    fgr = cp.array(fgr)
    pha = cp.array(pha)
    com = (fgr * pha + bgr * (1 - pha))*255 #运算得到最终画面
    com = com.reshape([3,720,1280]) #重复第一次的操作还原数组
    com = com.astype('int8')
    com = com.swapaxes(0,1).swapaxes(1,2)
    com = cp.asnumpy(com)
    out.write(com)
cap.release()
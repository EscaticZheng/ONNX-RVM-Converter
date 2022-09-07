# ONNX-RVM-Converter
Using cupy and onnx-runtime-gpu to boost the performance on **Robust Video Matting**[[link]](https://github.com/PeterL1n/RobustVideoMatting) for onnx.  
On my *i7-9700k* and *RTX3060* win10 computer, two video matting only takes *40% cpu* and *60% gpu*.  
    
2022/09/07 add two srcipts for inference on *cpu*.
    
**Time consuming**:  
~30S for converting a 30s video on HDD.  
~22s for converting a 30s video on SSD.  
2min for 30s video on an i9 cpu.
# Requirements
**Gpu**:  
FFmpeg  
Opencv-python  
Cuda and cudnn  
Numpy and cupy(For windows, you need to install the visual studio first, then run or re-run the cuda installer)  
Onnxruntime-gpu  
tqdm  
**Note:The package version depends on your computer.**  
**Cpu**:  
onnxruntime  
tqdm  
opencv-python  
numpy  
**Note:You'd better create a new env for install.**  
# Usage
1.Download my project zip or git clone. 
   
2.put your test video(avi) in the same directory and rename input.avi.  
  
3.run **onnx_gpu_infer.py** for FP16 inference or **onnx_gpu_infer_fp32.py** for FP32 inference(it takes twice the time)  
  
or run **onnx_cpu_infer.py** for FP16 inference or **onnx_cpu_infer_fp32.py** for FP32 inference  
  **cpu is better on fp32 than fp16**

4.Then look at the **com.avi** result.(If it's 0k means have some errors on your opencv) 
   
For further usage, just change the code whatever you like(e.g. background color, video format).  


# Opencv-Python-RVM Converter
Using cupy and onnx-runtime-gpu to boost the performance on **Robust Video Matting**[[link]](https://github.com/PeterL1n/RobustVideoMatting) for onnx.  
On my *i7-9700k* and *RTX3060* win10 computer, two video matting only takes *40% cpu* and *60% gpu*.  
**Time consuming**:~30S for converting a 30s video.
# Requirements
FFmpeg
Opencv-python 
Cuda and cudnn  
Numpy and cupy(For windows, you need to install the visual studio first, then run or re-run the cuda installer) 
Onnxruntime-gpu
**Note**:The package version depends on your computer.
# Usage
1.Download my project zip or git clone.  
2.put your test video(mp4) in the same directory and rename input.mp4.  
3.run **onnx_gpu_infer.py** for FP16 inference or **onnx_gpu_infer_fp32.py** for FP32 inference(it takes twice the time)  
4.Then look at the **com.avi** result.(If it's 0k means have some errors on your opencv)  
For further usage, just change the code whatever you like(e.g. background color, video format).  

## Hardware and Software Specs 
* Hardware:  
    * NVIDIA RTX 2060 Super  
* OS:  
    * Windows 10  
        
* Software:  
    * tensorflow-gpu-1.13.2 
    * Python = 3.7  
    * CUDA: 7.4    
    * cuDNN: 10  
  

## Virtual Environment Setup
A virtual environment is a tool that helps to keep dependencies required by different projects separate by creating isolated python virtual environments for them. Sort of like your house has kitchen, bedroom, bathroom... and each room is meant for each specific activity.  
1. Download and install [Anaconda](https://www.anaconda.com/products/individual) for windows    
2. Download this repo and unzip it 
3. On your search bar, type "Anaconda Prompt (anaconda3)" and right click on it and click "Run as Adminstrator"  
4. Create a VM (anyname you want) by type 
```Bash
conda create -n xxxx pip python=3.x 
```  
"xxxx" is the name of the new virtual environment and "x" is the corresponding python version. 
6. Activate this environment and update pip  
```Bash
activate xxxx
python -m pip install --upgrade pip 
```  
6. Install Tensorflow GPU version - This [website](https://www.tensorflow.org/install/source_windows ) gives you the tested build configurations for windows OS. For tensorflow-gpu-1.13.x version, the compatible Python version is 3.5-3.7, cuDNN is 7.4 and CUDA is 10. 

## CUDA and cuDNN installation 
[CUDA](https://developer.nvidia.com/cuda-toolkit) is a parallel computing platform and API that allows software to use certain types of GPU for general purpose. Make sure you select the compatible version.     
[cuDNN](https://developer.nvidia.com/cudnn-download-survey) is a GPU-accelerated library primitives for deep neural networks. Make sure you select the compatible version.
Configuration of CUDA and cuDNN should be automaticlly done, however, I recommend go to the windows system variable window to look it up.  
A restart of the system is recommended after the installation of CUDA and cuDNN. 

## TensorFlow API repo setup <br>
This part is already done for those who would like to keep using `tensorflow-gpu-1.13.x` version to train classifiers. However, if you want to use the most updated version of TensorFlow API, navigate to this [link](https://github.com/tensorflow/models). Download and unzip it. Noted I wrote some additional scripts and put some object detection models into the old code, please keep the [object_detection](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection) folder as it was. <br>

## TensorFlow Model Selection <br>
TensorFlow provides a bunch of object detection models in this [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Each model has their own advantages and disadvanteges in terms of specific application you choose. For this repo, We will use Faster-RCNN-Inception-V2 Model. It's one of the famous object detection architectures that uses CNN. This [article](https://towardsdatascience.com/faster-rcnn-object-detection-f865e5ed7fc4) explains how this architecture works in details.




























  

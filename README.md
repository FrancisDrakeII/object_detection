## Below is the hardware and software environment.  
Hardware:  
    1. NVIDIA RTX 2060 Super  

OS:  
        1. Windows 10  
        
Software:  
        1. tensorflow-gpu-1.13.2  
        2. Python = 3.7
        3. CUDA: 7.4  
        4. cuDNN: 10  
  

## Virtual Environment creation
A virtual environment is a tool that helps to keep dependencies required by different projects separate by creating isolated python virtual environments for them. Sort of like your house has kitchen, bedroom, bathroom... and each room is meant for each specific activity.  
1. Download and install Anaconda for windows https://www.anaconda.com/products/individual   
2. Download this repo and unzip it 
3. On your search bar, type "Anaconda Prompt (anaconda3)" and right click on it and click "Run as Adminstrator"  
4. Create a VM (anyname you want) by type ```conda create -n xxxx pip python=3.x ```  "xxxx" is the name of the new virtual environment and "x" is the corresponding python version. 
5. Activate this environment and update pip  
        ``` activate xxxx```  
        ``` python -m pip install --upgrade pip```  
6. Install Tensorflow GPU version - This [website](https://www.tensorflow.org/install/source_windows ) gives you the tested build configurations for windows OS. For tensorflow-gpu-1.13.x version, the compatible Python version is 3.5-3.7, cuDNN is 7.4 and CUDA is 10. 

## CUDA and cuDNN installation 
CUDA is a parallel computing platform and API that allows software to use certain types of GPU for general purpose. Here's the download site https://developer.nvidia.com/cuda-toolkit. Make sure you select the compatible version.   
cuDNN is a GPU-accelerated library primitives for deep neural networks. Here's the download site https://developer.nvidia.com/cudnn-download-survey Make sure you select the compatible version.


  

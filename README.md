## Hardware and Software Specs <br>  
* Hardware:  
    * 1. NVIDIA RTX 2060 Super  
* OS:  
    * 1. Windows 10  
        
* Software:  
    * 1. tensorflow-gpu-1.13.2 
    * 2. Python = 3.7  
    * 3. CUDA: 7.4    
    * 4. cuDNN: 10  
  

## Virtual Environment creation
A virtual environment is a tool that helps to keep dependencies required by different projects separate by creating isolated python virtual environments for them. Sort of like your house has kitchen, bedroom, bathroom... and each room is meant for each specific activity.  
1. Download and install [Anaconda](https://www.anaconda.com/products/individual) for windows    
2. Download this repo and unzip it 
3. On your search bar, type "Anaconda Prompt (anaconda3)" and right click on it and click "Run as Adminstrator"  
4. Create a VM (anyname you want) by type ```conda create -n xxxx pip python=3.x ```  "xxxx" is the name of the new virtual environment and "x" is the corresponding python version. 
5. Activate this environment and update pip  
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


  

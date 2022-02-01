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
3. On your search bar, type `Anaconda Prompt (anaconda3)` and right click on it and click "Run as Adminstrator"  
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
```Bash
pip install tensorflow-gpu-x.xx.x
```
x.xx.x is the specified version.

## CUDA and cuDNN installation 
[CUDA](https://developer.nvidia.com/cuda-toolkit) is a parallel computing platform and API that allows software to use certain types of GPU for general purpose. Make sure you select the compatible version.     
[cuDNN](https://developer.nvidia.com/cudnn-download-survey) is a GPU-accelerated library primitives for deep neural networks. Make sure you select the compatible version.
Configuration of CUDA and cuDNN should be automaticlly done, however, I recommend go to the windows system variable window to look it up.  
A restart of the system is recommended after the installation of CUDA and cuDNN. 

## TensorFlow API repo setup <br>
This part is already done for those who would like to keep using `tensorflow-gpu-1.13.x` version to train classifiers. However, if you want to use the most updated version of TensorFlow API, navigate to this [link](https://github.com/tensorflow/models). Download and unzip it. Noted I wrote some additional scripts and put some object detection models into the old code, please keep the [object_detection](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection) folder as it was. <br>

## TensorFlow Model Selection <br>
TensorFlow provides a bunch of object detection models in this [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Each model has their own advantages and disadvanteges in terms of specific application you choose. For this repo, We will use Faster-RCNN-Inception-V2 Model ([tf1_detection_zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)). It's one of the famous object detection architectures that uses CNN. This [article](https://towardsdatascience.com/faster-rcnn-object-detection-f865e5ed7fc4) explains how this architecture works in details.

## Training Object Detection Classifier <br>
* Go to `/object_detection/images/train` [folder](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection/images/train) and Go to `/object_detection/images/test` [folder](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection/images/test), delete everything except `labelImg.exe`. <br>
* Go to `/object_detection/images` [folder](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection/images), delete `test_labels.csv` and `train_labels.csv`. <br>
* Go to `/object_detection/training` [folder](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection/training), delete everything. <br>
* Go to `/object_detection/inference_graph` [folder](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection/inference_graph), delete everything. <br>

## VM setup continued...
* Reactivate your environment inside the Anaconda Prompt (anaconda3) terminal (remember run it as adminstrator). 
* Pip install the following packages
```Bash
conda install -c anaconda protobuf
pip install opencv-python
pip install pillow
pip install lxml
pip install Cython
pip install contextlib2
pip install jupyter
pip install matplotlib
pip install pandas
```
You might need more packages, if the system throws some error messages as `xxxx library is required or a higher version is needed`, just `pip install/uninstall xxxx` and install the compatible version of it. 
* Configure the PYTHONPATH environment variable on your local host
   * Right click `This PC`, scroll down to `Advanced system settings` and click on it, click on `Environment Variables`, In `System variables` section, click on `New...`, In `Variable name` part, type `PYTHONPATH`, in `Variable value` part, type `C:\xxxx\models`. Note `xxxx` is the file path of this repo on your local host. 
   * Repeat the above step for 2 times for `C:\xxxx\models\research` & `C:\xxx\models\research\slim`, both variable names are the same as `PYTHONPATH` above.
* Compile Protobufs
   * Compile the Protobufs files that used by TF to configure model and training parameters. If you're training on a Windows OS using GPU, please proceed with the following commands for protoc compilation. Basically, we call out every `.proto` file in `object_detection/protos` [folder](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection/protos) and create a `xxxx_pb2.py` file from every `xxxx.proto` file in this folder.
 ```Bash
 protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
 ```
  * Note if if you get an error saying `ImportError: cannot import name 'xxxx_pb2'`, update the protoc cammand to include the missing .proto files
  * Run the following command from `C:\xxxx\models\research` directory:
```Bash
python setup.py build
python setup.py install
```





















  

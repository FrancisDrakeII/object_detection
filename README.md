## Hardware and Software Specs 
Following is the list of the hardware, operation system, and software environment I used when I implement this project. These are for reference only, you can choose any GPU, systems you want if you want to recreate this project. However, you do need to pay attention to the software packages (especially the compatibility between them) because they're constantly getting updated by publishers.
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
* Download and install [Anaconda](https://www.anaconda.com/products/individual) for windows    
* Download this repo and unzip it (or git clone)
* On your search bar, type `Anaconda Prompt (anaconda3)` and right click on it and click "Run as Adminstrator"  
* Create a VM (anyname you want) by type 
```Bash
conda create -n xxxx pip python=3.x 
```  
"xxxx" is the name of the new virtual environment and "x" is the corresponding python version. (Please pay attention to the compatibility between tensorflow and Python as listed [here](https://www.tensorflow.org/install/source_windows)). For tensorflow-gpu-1.13.x version, the compatible Python version is 3.5-3.7, cuDNN is 7.4 and CUDA is 10. <br>    
* Activate this environment and update pip  
```Bash
activate xxxx
python -m pip install --upgrade pip 
```  
* Install Tensorflow GPU version - This [website](https://www.tensorflow.org/install/source_windows ) gives you the tested build configurations for windows OS.  
```Bash
pip install tensorflow-gpu-x.xx.x
```
x.xx.x is the specified version.

## CUDA and cuDNN installation 
[CUDA](https://developer.nvidia.com/cuda-toolkit) is a parallel computing platform and API that allows software to use certain types of GPU for general purpose. Make sure you select the compatible version.     
[cuDNN](https://developer.nvidia.com/cudnn-download-survey) is a GPU-accelerated library primitives for deep neural networks. Make sure you select the compatible version.
Again, pay attention to the [compatibility](https://www.tensorflow.org/install/source_windows) between Python, CUDA and cuDNN. 
Configuration of CUDA and cuDNN should be automaticlly done, however, I recommend go to the windows system variable window to look it up.  
A restart of the system is recommended at this stage after the installation of CUDA and cuDNN. 

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

## Data Collecting and Labelling
* Once you gathered enough images (hundreds to thousands) you need for training, allocate 20% of them to `/object_detection/images/test` folder and 80% of them to `/object_detection/images/train` folder.
* Label the images using the `LabelImg.exe` by opening it, navigating to the `images/train` and `images/test` folder. 
* To generate training data, navigate to the `\object_detection folder` folder inside the anaconda prompt, then type
```Bash
python xml_to_csv.py
``` 
This script converts label data inside each image file from xml format to csv, which will be used in the later training process. It will create a [train_labels.csv](https://github.com/FrancisDrakeII/object_detection/blob/main/models/research/object_detection/images/test_labels.csv) and [test_label.csv](https://github.com/FrancisDrakeII/object_detection/blob/main/models/research/object_detection/images/train_labels.csv) file in the  [/object_detection/images](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection/images) folder. <br>
* Open [generate_tfrecord.py](https://github.com/FrancisDrakeII/object_detection/blob/main/models/research/object_detection/generate_tfrecord.py) and move to line 36 where the ```class_text_to_int``` function located at, change the name of the label to your defined label. <br>
* Navigate to `/object_detection` folder inside command terminal and issue the folling command to generate the [train.record](https://github.com/FrancisDrakeII/object_detection/blob/main/models/research/object_detection/train.record) and [test.record](https://github.com/FrancisDrakeII/object_detection/blob/main/models/research/object_detection/test.record) file.
```Bash
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```
These will be used to train the new object detection classifier. <br>

## Labelling and Training Configuration
* Go to `/object_detection/training` [folder](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection/training), create a file and save it as `labelmap.pbtxt`. Then open this file and type the following:
```Bash
item {
  id: 1
  name: 'xxx'
}

item {
  id: 2
  name: 'xxx'
}

.....
```
Note the name should match with the previous label name you created. <br>
* Training pipeline defines which model and what parameters will be used for training. Like mentioned above, this tutorial will use faster_rcnn_inception_v2 models. Navigate to `/object_detection/samples/configs` [folder](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection/samples/configs) and copy the `faster_rcnn_inception_v2_pets.config` file into the [/object_detection/training](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection/training) folder
* Open this file under text editor option, Make the following changes:
   * Line 9, change `num_classes` to the number of different objects you wants the classifier to detect. In my example, I want to detect "white","yellow","pod","bud", so `num_classes` would be 4.
   * Line 106, change `fine_tune_checkpoint` to `"C:/xxxx/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"`
   * Line 123, change `input_path` to `"C:/xxxx/models/research/object_detection/train.record"`
   * Line 125, change `label_map_path` to `"C:/xxxx/models/research/object_detection/training/labelmap.pbtxt"`
   * Line 130, change `num_examples` to the number of images in `/object_detection/images/test` [directory](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection/images/test). This number should be the total number of files in this (folder-1)/2. 
   * Line 135, change `input_path` to `"C:/xxxx/models/research/object_detection/test.record"`
   * Line 137, change `label_map_path` to `"C:/xxxx/models/research/object_detection/training/labelmap.pbtxt"` 
 * Save the changes when exiting 
 
 ## Training Process
 If you're using __tensorflow-2.x.x__, *TensorFlow* deprecated the `train.py` file and replaced with `model_main.py` file, please change the command accordingly. <br>
 * Navigate to /object_detection folder in command window, issue the following command to begin training: 
```Bash
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```
* Initialization process may take up to 30 seconds to begin, if you saw some errors, I first suggest you to copy the error code and Google it. Most of the solutions should be there. If you still have problems, please reach me out via my email and give me the error message/log: zihuanteng@gmail.com. 
* During training process, window will prompt the steps you are, loss in real time, depends on your training data size, model you chose, training time will vary. For Faster-RCNN-Inception-V2 model, I recommend stop the training process when loss consistently drops below 0.05. For light-weighted model such as the MobileNet-SSD series (usually implemented on Raspberry Pi/NVIDIA Jetson), I recommend stop the training when the loss consistently drops below 2.
* To view the progress of training in details, Open a new Anaconda Prompt window as Adminstrator, activate the VM envirnment, and navigate to the ` C:\tensorflow1\models\research\object_detection` [directory](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection), and issue the command:
```Bash
tensorboard --logdir=training
```
Now you can visualize the training progress via graphs in real time. 
* To stop the training, simply pressing `Ctrl+C` in the command window. (Don't close the window yet!!!)
* To export the training model, make sure you're in `/object_detection` [folder](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection) inside command terminal. Then issue the following command:
```Bash
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```
The XXXX is the highest number .ckpt file in the [training folder](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection/training).

## Implement the Trained Object Detection Classifier
__Before moving onto the implementation, I just want to remind again of the growing ethical/moral concerns using AI technology, no matter what kind of instances you're trying to achieve using the machine learning, please make sure it follows the moral conduct.__
* At this stage, you can write your own python script under `/object_detection` [folder](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection). However, if you want to test it out using my code, feel free to do so.
  * Modify the `NUM_CLASSES` (Line 33) variable in the `Object_detection_webcam.py` script to the number you defined.
  * Run this script by issuing the following command (Make sure you navigate to `/object_detection` [folder](https://github.com/FrancisDrakeII/object_detection/tree/main/models/research/object_detection) in command prompt and your computer has a camera): 
```Bash
python Object_detection_webcam.py
```
Now you can see the objects it's detected in real time!

## Acknowledgement 
I would like to express my gratitude and respect to the following people for your help and support in this project. Through this project, I have a clear career plan in the direction of SDE/robotics：<br> 

Dr. William Eisenstadt <br>
Dr. Ian Small <br>
Dr. Rebecca Barocco <br> 













  

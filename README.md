# Emotions Detections
This project uses Pytorch Framework for human face emotions detection. The project can run on images from the test set as well on the any video or live camera. The code first extracts the face from an image and then feed the face into the model for classification. The code implements model building and training, model evaluation on test set and loading data using data loaders.


# How to run the code
First we need to install the required libraries. If you have GPU installed on your machine, then you better install Pytorch and Torchvision from the link given below according to your CUDA and cuDNN versions.
    https://pytorch.org/get-started/locally/

Then install the additional libraries using the following command:

    pip install -r requirements.txt

Finally, run train.py for training the model and inference.py for running the model on real data. There are two function calls in inference.py for images and for webcam or video.

### Note
While training, the dataloader loads the data from the dataset path which has been designed for specific directory structure. The structure should be:

    dataset path -->
                train -->
                    image_1.jpg, image_2.jpg, .... n
                test -->
                    image_1.jpg, image_2.jpg, .... n
                labels --> 
                    train.txt
                    test.txt

## CONFIG.py
This file is to specify parameters which are being used in the entire project. You must alter those parameters according to your dataset and other paths. For Example, in training parameters, you'd need to modify the number of classes, number of epochs, batch size and so on.

If RESUME is true, the code will first try to load the checkpoint specified as CHECKPOINT_PATH. All checkpoints during training will be saved in CHECKPOINT_DIR.

We need to must specify the classes.txt file as PATH_TO_CLASSES_TXT which is then used in inference. The classes.txt files contains the names of classes based on line number e.g. class 0 will be mapped with the name at the first line in classes.txt and so on.

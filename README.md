# Chest-X-ray-Images-Pneumonia-Classifier
Using a convolutional neural network to predict whether an individual has pneumonia from a chest x-ray image.

This mini-project utilizes Fastai and PyTorch to develop and fine-tune a pretrained resnet50 model. The data was downloaded from 
the kaggle dataset "Chest X-Ray Images (Pneumonia)" found at this link: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia. 
I downloaded this model to my google drive first (which one would have to do if using my code). 

## Two models were trained:
1) A resnet50 pretrained on imagenet was instantiated, wherein I then additionally trained a few extra fully connected layers.
  The model metrics measured on a test set:
    a) Recall -> 98.4%
    b) Accuracy -> 91.0%
    c) Precision -> 77.4%
    
 2) This model was a continuation of the previous model; the pretrained networks parameters were unfrozen to enable training of all
    the weights. This model ended up performing slightly worse than the prior model (1 extra misclassified example).
   The model metrics measured on a test set:
    a) Recall -> 98.4%
    b) Accuracy -> 90.9%
    c) Precision -> 77.0%
   
 Overall, this mini-project shows the capabilities of transfer learning in computer vision applications to achieve state-of-the-art results
 with relatively minimal effort.

# ColonCancerDetection
This project uses a dataset on image detection for colon cancer using python, keras and tensorflow to create a CNN.

the data came from Roboflow - polyp_detection.v3i.yolov5pytorch is the zip file you can get from them.

After reading a paper I am going to try a U-NN. It is interesting because it looks similar to a tranformer model with an encode and a decoder. I have also been looking at visual Transformer models that only use the encoder side of the model.

I ran the U NN on a RTX 3050 and it only took 14 minutes to run one epoch. Although the masks are not perfect it was able to identify where the polyps are the shape is jsut a square instead of the shape of the polyp. I assume if I run more epoch this will perform better.

2,266 training images and 488 validation images
Training phase: 284 batches
Final training loss: 0.4080

61 validation batches (100%) completed in about 2.5 minutes
Final validation loss: 0.6223

Model Performance Metrics:
Accuracy: 0.7473
Precision: 0.7444
Recall: 0.0083
F1 Score: 0.0164
AUC: 0.7914

![roc_curve](https://github.com/user-attachments/assets/1035b54a-7e2b-4cf1-b0fd-444601422b7d)

![result_1d80203b-d183-4aa8-904d-ea3111560e20_jpg rf 2b703d2c8835dffffc35016f3fd5101f](https://github.com/user-attachments/assets/93b05727-1253-441c-9360-8213cc92610f)








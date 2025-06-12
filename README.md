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




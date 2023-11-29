For this task, I decided to use LoFTR, a model based on Transformers (https://arxiv.org/pdf/2104.00680.pdf), but initially, data needs to be obtained. Satellite images have high resolution, making it challenging to use them in this algorithm. Therefore, I started by dividing the images into fragments.

LoFTR works with grayscale images. To use LoFTR, the kornia library was employed, along with pretrained weights.

File descriptions:

task2.py: Divides the satellite image into fragments.
task2_match.py: Compares two images and outputs matched keypoints

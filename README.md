# FlexICL

This is the repository of paper [FlexICL: A Flexible Visual In-Context Learning Framework for Elbow and Wrist Ultrasound Segmentation](https://ieeexplore.ieee.org/abstract/document/11494781)

Thanks to Zhenda Xie: code under "models" folder are mainly from Xie's paper: SimMIM: A Simple Framework for Masked Image Modeling, https://github.com/microsoft/SimMIM

## Introduction

FlexICL is a flexible framework combining visual in-context learning and masked image modeling for ultrasound image segmentation. We considered image segmentation task as image inpainting, in which the model is shown an example of image input and output, and asked to paint the output of a new image. To achieve this, we concatenated a support pair consisting of image/mask with another query pair consisting of image/(mask). Random masking was added to the entire concatenated image before feeding the image to the model.

To further improve generalization and reduce reliance on memorizing image appearances, we introduce a context-aware enhancement technique for processing the support-query pairs. This strategy encourages the model to focus more on contextual and structural relationships between support and query images, rather than directly memorizing specific visual patterns.

## Getting started

Please prepare the csv file including image filenames before running the code.

To Train the model:
```bash
python train.py \
--epochs 1200 \
--dataset_path '/dataset_path/' \
--train_csv 'train_csv_file.csv' \
--save_dir './results/' \
--frac_ratio 0.01 \
--shuffle_training \
--random_masking \
--aug_threshold 0.5 \
--device_gpu 'cuda:0' \
```

Model inference:
```bash
python /home/yuyue2/code/SimICL/inference.py \
--dataset_path '/dataset_path/' \
--train_csv 'train_csv_file.csv' \
--test_csv 'test_csv_file.csv' \
--save_dir './results/' \
--model_name 'latest_model.pth' \
--random_masking \
--device_gpu 'cuda:0' \
--saved_image_folder 'test_phase/'
```



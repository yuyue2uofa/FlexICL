# FlexICL

This is the repository of paper [FlexICL: A Flexible Visual In-Context Learning Framework for Elbow and Wrist Ultrasound Segmentation](https://ieeexplore.ieee.org/abstract/document/11494781)

Thanks to Zhenda Xie: code under "models" folder are mainly from Xie's paper: SimMIM: A Simple Framework for Masked Image Modeling, https://github.com/microsoft/SimMIM


To Train the model:

python train.py \
--epochs 1200 \
--dataset_path '/dataset_path/' \
--train_csv 'train_csv_file.csv' \
--save_dir './results/' \
--frac_ratio 0.01 \
--shuffle_training \
--random_masking \
--aug_threshold 0.5 \
--device_gpu 'cuda:0' 

Model inference:

python /home/yuyue2/code/SimICL/inference.py \
--dataset_path '/dataset_path/' \
--train_csv 'train_csv_file.csv' \
--test_csv 'test_csv_file.csv' \
--save_dir './results/' \
--model_name 'latest_model.pth' \
--random_masking \
--device_gpu 'cuda:0' \
--saved_image_folder 'test_phase/' 




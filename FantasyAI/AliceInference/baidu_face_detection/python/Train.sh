
export CUDA_VISIBLE_DEVICES=0

#pip install scikit-image
#pip install matplotlib
#pip install imgaug
#pip install expt-client -i http://pip.baidu.com/root/baidu/+simple/ --trusted pip.baidu.com


#python train_steps_caffe.py

#python -m paddle.distributed.launch eval/predict_images_ecloud.py
#python -m paddle.distributed.launch eval/predict_images.py
#python -m paddle.distributed.launch eval/predict_images_newface.py

python -m paddle.distributed.launch ./predict_images_newface_ecloud.py


文件目录：

```
.
├── bbox_utils.py
├── config.ini
├── darknet.py
├── dataset.py
├── datasets
├── each_node_bak.sh
├── each_node.sh
├── evalScript
├── hs_err_pid4165.log
├── iou_loss.py
├── layers.py
├── lmdb_util.py
├── log
├── logger.py
├── logs
├── map_utils.py
├── metrics.py
├── mobilenet_2d4f5904.py
├── mobilenet_2d96.py
├── mobilenet_50M224.py
├── mobilenet_792bcf52.py
├── mobilenet_b0b5e552.py
├── mobilenet_c7230302.py
├── mobilenet_v3.py
├── modify_quant_layers.py
├── operators_beifen.py
├── operators_MergeNew.py
├── operators.py
├── op_helper.py
├── ops.py
├── post_process.py
├── PPYoloMobileNetV3.py
├── predict_images_cloud.py
├── predict_images.py
├── predict_images_yu1.py
├── predict_images_yu2.py
├── predict_images_yu_fuse.py
├── predict_images_yu_nofuse.py
├── predict_images_yu_pd222.py
├── predict_images_yu_pd222_QA.py
├── predict_images_yu_test.py
├── predict_images_yu_xinsuanyiti_prune.py
├── prepare_lmdb.py
├── reader.py
├── README.md
├── run_cq.sh
├── run-gpu.sh
├── run.sh
├── shape_spec.py
├── stats.py
├── test_yolov3_loss.py
├── train_face1_cloud_beifen.py
├── train_face1_cloud_beifenzuixin.py
├── train_face1_cloud.py                       # 云端训练脚本
├── train_face1_local.py
├── train_face1.py
├── train_facedetect_cloud.sh
├── train_model.sh
├── train.py
├── train_utils.py
├── vehicle_pdc_facedetect_1.py
├── vehicle_pdc_facedetect.py
├── vehicle_pdc_hand.py
├── vehicle_pdc.py
├── x2paddle_code_2d4f5904_prune50.py
├── x2paddle_code_2d4f5904.py
├── yolo_fpn.py
├── yolo_head_one.py
├── yolo_head.py
└── yolo_loss.py

```





# 云端训练

```powershell
sh run.sh
```





# 本地训练

安装依赖环境

```
pip install paddlepaddle==2.3.0 -i https://mirror.baidu.com/pypi/simple
pip install paddleslim==2.3.0 -i https://mirror.baidu.com/pypi/simple

```



# 训练脚本

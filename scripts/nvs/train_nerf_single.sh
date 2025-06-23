ns-train nerfacto --pipeline.model.camera-optimizer.mode off \
--data $1 --vis tensorboard \
colmap --load_3D_points False --images_path images_stride_10 --colmap_path our_gt_to_user
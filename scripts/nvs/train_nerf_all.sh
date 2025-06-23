for scene_id in 51 98 129 131 154
do
  sbatch train_single_sbatch.sh "train_ours_nerf_single.sh" "./data/new_scanning_user_view_${scene_id}/"
done


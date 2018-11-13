
python test_net_with_rois.py --dataset pascal_voc --net res101 --checksession 1 --checkepoch 10 --checkpoint 2504 --load_dir save_dir   --neighbor_move 0.4 --cag --cuda  --reduce_dimension 256   --reg_neighbor --reg_reduce_d  --cls_neighbor --sigma_geometry 0.3 --cls_alpha_option 2 --alpha_same_with_beta


python test_net_with_rois_readresults.py --dataset pascal_voc --net res101 --checksession 1 --checkepoch 10 --checkpoint 2504 --load_dir save_dir   --neighbor_move 0.4 --cag --cuda  --reduce_dimension 256   --reg_neighbor --reg_reduce_d  --cls_neighbor --sigma_geometry 0.3 --cls_alpha_option 2 --alpha_same_with_beta 




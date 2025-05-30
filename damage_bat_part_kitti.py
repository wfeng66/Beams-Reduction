import os, shutil

root_dir = '/home/lab/feng/OpenPCDet'
damaged_beams = ["0,16,32,62", "0,8,16,24,32,40,48,62", "0,4,8,12,16,20,24,28,32,34,36,40,44,48,52,56,60,62", "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62", "0,2,4,6,8,10,11,12,14,16,18,20,21,22,24,26,28,30,31,32,34,36,38,40,41,42,44,46,48,50,51,52,54,56,58,60,61,62", "0,2,4,5,6,8,10,11,12,14,15,16,18,20,21,22,24,25,26,28,30,31,32,34,35,36,38,40,41,42,44,45,46,48,50,51,52,54,55,56,58,60,61,62", "0,1,2,4,5,6,8,9,10,11,12,14,15,16,18,20,21,22,24,25,26,27,28,30,31,32,34,35,36,37,38,40,41,42,43,44,45,46,48,49,50,51,52,54,55,56,58,59,60,61,62"]

for dmg_beam in damaged_beams:
    print("Running with the number of beams reduction: ", dmg_beam)
    print("Begin to recover the data!!!")
    
    # print(os.system('pwd'))
    # print('cd data/bak')
    # os.chdir('data/bak')
    # print(os.system('pwd'))
    # os.system('cp -r kitti ../.')
    if os.path.exists(root_dir+'/data/kitti'):
        shutil.rmtree(root_dir+'/data/kitti')
    shutil.copytree(root_dir+'/data/bak/kitti', root_dir+'/data/kitti')
    # os.chdir(root_dir+'/data/kitti')
    # os.system('rm *.pkl')
    # os.remove(root_dir+'/data/kitti/*.pkl')
    # os.system('rm -r gt_database')
    # shutil.rmtree(root_dir+'/data/kitti/gt_database')
    # os.chdir(root_dir)
    # os.system('rm -rf output/kitti_models/second/default/*')
    os.remove(root_dir+'/output/kitti_models/second/default/*.*')
    
    print("Running beams reduction: ", dmg_beam)
    os.system('python damage_beams.py data/kitti 4 64 '+ dmg_beam)
    
    print("Recreate the GT dataset and info: ", dmg_beam)
    os.system('python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml')
    
    os.system('cd tools')
    print("Evaluate reducted beams dataset: ", dmg_beam)
    os.system("python train.py  --cfg_file cfgs/kitti_models/PartA2.yaml  --batch_size 16   --pretrained_model pretrained/PartA2_7940.pth --num_epochs_to_eval 1 --worker 16 --train False --dataset_version " + dmg_beam)


print("All done.")
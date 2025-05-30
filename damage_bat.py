import os, shutil

root_dir = '/home/lab/feng/OpenPCDet'
damaged_beams = ["2,28", "2,8,16,28", "2,8,12,16,24,28"]

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
    os.system('python damage_beams.py data/kitti 4 32 '+ dmg_beam)
    
    print("Recreate the GT dataset and info: ", dmg_beam)
    os.system('python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml')
    
    os.system('cd tools')
    print("Evaluate reducted beams dataset: ", dmg_beam)
    os.system("python train.py  --cfg_file cfgs/kitti_models/second.yaml  --batch_size 16   --pretrained_model pretrained/second.pth --num_epochs_to_eval 1 --worker 16 --train False --dataset_version " + dmg_beam)


print("All done.")
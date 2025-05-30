###############################
# This scrtipt is used to randomly select damaged beams and
# calculate the mean of performance
#################################

import os, shutil, argparse, random

def recover(root, dataset):
    print("Begin to recover the data!!!")
    
    if os.path.exists(root+'/data/' + dataset):
        shutil.rmtree(root+'/data/' + dataset)
    shutil.copytree(root+'/data/bak/' + dataset, root+'/data/' + dataset)

    # os.remove(root+'/output/kitti_models/second/default/*.*')    


def damageNevaluate(root, dmg_beam, yaml, pretrained, total_beams, dataset):
    os.chdir(root)    
    n_features = 4 if dataset == 'kitti' else 5
    print("Running beams reduction: ", dmg_beam)
    dmg_str = 'python damage_beams.py data/' + dataset + ' ' + str(n_features) + ' ' + total_beams + ' ' + dmg_beam
    os.system(dmg_str)
    # os.system('python damage_beams.py data/kitti 4 32 '+ dmg_beam)
    
    print("Recreate the GT dataset and info: ", dmg_beam)
    info_str = 'python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml' if dataset == 'kitti' \
        else "python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval"
    os.system(info_str)
    # os.system('python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml')
    
    os.chdir(root+'/tools')
    print("Evaluate reducted beams dataset: ", dmg_beam)
    train_str = "python train.py  --cfg_file cfgs/" + dataset + "_models/" + yaml + "  --batch_size 16   --pretrained_model pretrained/" + \
        pretrained +  "  --num_epochs_to_eval 1 --worker 16 --train False --dataset_version " + dmg_beam
    os.system(train_str)
    # os.system("python train.py  --cfg_file cfgs/kitti_models/second.yaml  --batch_size 16   --pretrained_model pretrained/second.pth --num_epochs_to_eval 1 --worker 16 --train False --dataset_version " + dmg_beam)




def main(root, dataset,total_beams, iter, yaml, pretrained):

    for n_dmg in range(2, int(total_beams), 2):
        for _ in range(int(iter)):      # iterate iter times for each number of beams to be damaged
            # randomly select damaged beams number
            rand_beams = random.sample(range(int(total_beams)), n_dmg)
            dmg_beam = ",".join(map(str, rand_beams))
            
            print("Running with the number of beams reduction: ", n_dmg, " : ", str(_), '/'+iter)
            recover(root, dataset)
            
            damageNevaluate(root, dmg_beam, yaml, pretrained, total_beams, dataset)
            
    print("All done.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--root', type=str, default='/home/lab/feng/OpenPCDet', help='the root path of OpenPCDet')
    parser.add_argument('--dataset', type=str, default='kitti', help='the dataset use for evaluation')
    parser.add_argument('--total_beams', type=str, default='64', help='the total number of beams of the sensor')
    parser.add_argument('--iter', type=str, default='10', help='number of times to iterate and get the mean')
    parser.add_argument('--yaml', type=str, default='second.yaml', help='the name of yaml file')
    parser.add_argument('--pretrained', type=str, default='second.pth', help='the name of pretrained model')
    args = parser.parse_args()
    
    main(root=args.root, dataset=args.dataset, total_beams=args.total_beams, iter=args.iter, yaml=args.yaml, pretrained=args.pretrained)
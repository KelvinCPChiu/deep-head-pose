import sys, os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import datasets
import hopenet
import utils

import torch.utils.model_zoo as model_zoo

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=5, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshot.', default = '', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=0.001, type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)

    args = parser.parse_args()
    return args


def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == '__main__':

    # Change to .to(device) will give the flexibility to the work on either GPU/CPU environment
    # Especially, when doing code checking, sometime CPU woul be more transparent.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print 'It is running with', device, '.'

    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    num_bins = 66

    if not os.path.exists('output/snapshot'):
        os.makedirs('output/snapshot')

    # ResNet50 structure

    # The number of bins could be changed along with the idx 66 below in the training function.

    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins)

    if args.snapshot == '':
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    else:
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)
        print 'Snapshot has been loaded.'

    print 'Loading data.'

    transformations = transforms.Compose([transforms.Resize(240),
    transforms.RandomCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets.Pose_300W_LP_random_ds(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Synhead':
        pose_dataset = datasets.Synhead(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(args.data_dir, args.filename_list, transformations)
    else:
        print 'Error: not a valid dataset name'
        sys.exit()

    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    reg_criterion = nn.MSELoss().to(device)
    # Regression loss coefficient
    alpha = args.alpha

    # In Pytorch >1.0, it is better to define the dimension of softmax being done.

    softmax = nn.Softmax(dim=1).to(device)

    idx_tensor = [idx for idx in xrange(num_bins)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)

    # We do not need to write the optimizer in this way.

    optimizer = torch.optim.Adam(model.parameters(), lr=10**-5, eps=10**-8)

    #upp = torch.tensor([66]).to(device)
    #ddo = torch.tensor([-1]).to(device)

    #test_net = test_hope_net()
    print 'Ready to train network.'
    for epoch in range(num_epochs):
        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = images.to(device)

            # Binned labels
            label_yaw = labels[:, 0].to(device)
            label_pitch = labels[:, 1].to(device)
            label_roll = labels[:, 2].to(device)

            # Continuous labels
            label_yaw_cont = cont_labels[:, 0].to(device)
            label_pitch_cont = cont_labels[:, 1].to(device)
            label_roll_cont = cont_labels[:, 2].to(device)

            # Forward pass
            yaw, pitch, roll = model(images)

            #tuple = (label_yaw, label_pitch, label_roll)
            #for data in tuple:
            #    test_u = torch.any(torch.gt(data, upp))
            #    test_d = torch.any(torch.lt(data, ddo))
            #    test = torch.any(torch.eq(data, 0))
            #    test_1 = torch.any(torch.eq(data, 1))
            #    if test_d or test_u:
            #        print 'Mini-batch Index :', i, name
            #        print(data)
            #   if test or test_1:
            #        print(data)

            # Cross entropy loss
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            # MSE loss
            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq)
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f'
                        %(epoch+1, num_epochs, i+1, len(pose_dataset)//batch_size,
                            loss_yaw.item(), loss_pitch.item(), loss_roll.item()))
                #with torch.no_grad():
                #    test_network()

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print 'Taking snapshot...'
            torch.save(model.state_dict(),
            'output/snapshot/' + args.output_string + '_epoch_'+ str(epoch+1) + '.pkl')


class test_hope_net(object):
    """Adding Testing for Early Stopping Implementation."""
    def __init__(self, data_dir, transformation, device, num_bin):

        args.data_dir = data_dir
        args.filename_list = data_dir
        args.dataset = 'AFLW2000'
        self.transformations = transformation
        self.device = device
        self.num_bin = num_bin

        idx_tensor = [idx for idx in xrange(67)]
        self.idx_tensor = torch.FloatTensor(idx_tensor).to(self.device)
        if args.dataset == 'Pose_300W_LP':
            pose_dataset = datasets.Pose_300W_LP(args.data_dir, args.filename_list, self.transformations)
        elif args.dataset == 'Pose_300W_LP_random_ds':
            pose_dataset = datasets.Pose_300W_LP_random_ds(args.data_dir, args.filename_list, self.transformations)
        elif args.dataset == 'AFLW2000':
            pose_dataset = datasets.AFLW2000(args.data_dir, args.filename_list, self.transformations)
        elif args.dataset == 'AFLW2000_ds':
            pose_dataset = datasets.AFLW2000_ds(args.data_dir, args.filename_list, self.transformations)
        elif args.dataset == 'BIWI':
            pose_dataset = datasets.BIWI(args.data_dir, args.filename_list, self.transformations)
        elif args.dataset == 'AFLW':
            pose_dataset = datasets.AFLW(args.data_dir, args.filename_list, self.transformations)
        elif args.dataset == 'AFLW_aug':
            pose_dataset = datasets.AFLW_aug(args.data_dir, args.filename_list, self.transformations)
        elif args.dataset == 'AFW':
            pose_dataset = datasets.AFW(args.data_dir, args.filename_list, self.transformations)
        else:
            print 'Error: not a valid dataset name'
            sys.exit()

        self.test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=2)

    def test_network(self):

        print 'Ready to test network.'

        total = 0

        yaw_error = .0
        pitch_error = .0
        roll_error = .0

        l1loss = torch.nn.L1Loss(size_average=False)

        for i, (images, labels, cont_labels, name) in enumerate(self.test_loader):
            images = Variable(images).cuda(gpu)
            total += cont_labels.size(0)

            label_yaw = cont_labels[:,0].float()
            label_pitch = cont_labels[:,1].float()
            label_roll = cont_labels[:,2].float()

            yaw, pitch, roll = model(images)

            # Binned predictions
            _, yaw_bpred = torch.max(yaw.data, 1)
            _, pitch_bpred = torch.max(pitch.data, 1)
            _, roll_bpred = torch.max(roll.data, 1)

            # Continuous predictions
            yaw_predicted = utils.softmax_temperature(yaw.data, 1)
            pitch_predicted = utils.softmax_temperature(pitch.data, 1)
            roll_predicted = utils.softmax_temperature(roll.data, 1)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

            # Mean absolute error
            yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw))
            pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch))
            roll_error += torch.sum(torch.abs(roll_predicted - label_roll))

            # Save first image in batch with pose cube or axis.
            if args.save_viz:
                name = name[0]
                if args.dataset == 'BIWI':
                    cv2_img = cv2.imread(os.path.join(args.data_dir, name + '_rgb.png'))
                else:
                    cv2_img = cv2.imread(os.path.join(args.data_dir, name + '.jpg'))
                if args.batch_size == 1:
                    error_string = 'y %.2f, p %.2f, r %.2f' % \
                                   (torch.sum(torch.abs(yaw_predicted - label_yaw)),
                                    torch.sum(torch.abs(pitch_predicted - label_pitch)),
                                    torch.sum(torch.abs(roll_predicted - label_roll)))
                    cv2.putText(cv2_img, error_string, (30, cv2_img.shape[0]- 30), fontFace=1, fontScale=1, color=(0,0,255), thickness=2)
                # utils.plot_pose_cube(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], size=100)
                utils.draw_axis(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], tdx = 200, tdy= 200, size=100)
                cv2.imwrite(os.path.join('output/images', name + '.jpg'), cv2_img)

        print('Test error in degrees of the model on the ' + str(total) +
                ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f' %
              (yaw_error / total,
               pitch_error / total,
               roll_error / total))

        return yaw_error, pitch_error, roll_error

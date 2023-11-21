import argparse
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from ClassifierModel import ClassifierModel
from KittiAnchors import Anchors
from KittiDataset import KittiDataset
from ROIBatchDataset import ROIBatchDataset


def strip_ROIs(class_ID, label_list):
    ROIs = []
    for i in range(len(label_list)):
        ROI = label_list[i]
        if ROI[1] == class_ID:
            pt1 = (int(ROI[3]),int(ROI[2]))
            pt2 = (int(ROI[5]), int(ROI[4]))
            ROIs += [(pt1,pt2)]
    return ROIs


def calc_IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][1], boxB[0][1])
    yA = max(boxA[0][0], boxB[0][0])
    xB = min(boxA[1][1], boxB[1][1])
    yB = min(boxA[1][0], boxB[1][0])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[1][1] - boxA[0][1] + 1) * (boxA[1][0] - boxA[0][0] + 1)
    boxBArea = (boxB[1][1] - boxB[0][1] + 1) * (boxB[1][0] - boxB[0][0] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def calc_max_IoU(ROI, ROI_list):
    max_IoU = 0
    for i in range(len(ROI_list)):
        max_IoU = max(max_IoU, calc_IoU(ROI, ROI_list[i]))
    return max_IoU


def data_transform():
    transform_list = [
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def swap_box_xy(box):
    pt1 = (box[0][1], box[0][0])
    pt2 = (box[1][1], box[1][0])
    return pt1, pt2


if __name__ == '__main__':

    device = torch.device('cpu')

    # ----- set up argument parser for command line inputs ----- #
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-input_dir', metavar='input_dir', type=str, help='input dir (./)')
    argParser.add_argument('-model_dir', metavar='model_dir', type=str, help='model dir (./)')
    argParser.add_argument('-iou', metavar='iou_threshold', type=float, help='0.02')
    argParser.add_argument('-max_images', metavar='max_images', type=int, help='10')
    argParser.add_argument('-display', metavar='display', type=str, help='[y/N]')
    argParser.add_argument('-verbose', metavar='verbose', type=str, help='[y/N]')
    argParser.add_argument('-cuda', metavar='cuda', type=str, help='[y/N]')

    # ----- get args from the arg parser ----- #
    args = argParser.parse_args()
    input_dir = Path(args.input_dir)
    model_dir = Path(args.model_dir)
    IoU_threshold = float(args.iou)
    max_data_items = int(args.max_images)
    show_images = str(args.display).lower() == 'y'
    verbose = str(args.verbose).lower() == "y"
    use_cuda = str(args.cuda).lower() == 'y'

    print("CudaIsAvailable: {}, UseCuda: {}".format(torch.cuda.is_available(), use_cuda))
    if torch.cuda.is_available() and use_cuda:
        print('using cuda ...')
        device = torch.device('cuda')
    else:
        print('using cpu ...')

    # ----- initialize model ----- #
    model = ClassifierModel()
    model.load_state_dict(torch.load(model_dir))
    model.to(device=device)
    model.eval()
    print('model loaded OK!')

    # ----- initialize dataset ----- #
    transform = data_transform()
    dataset = KittiDataset(input_dir, training=False)
    anchors = Anchors()

    # ----- iterate through the dataset ----- #
    print("{} starting test...".format(datetime.now()))
    i = 0
    predicted_ious = []
    start_time = time.time()

    for item in enumerate(dataset):
        if verbose:
            print("item #", i+1)
        idx = item[0]
        image = item[1][0]
        label = item[1][1]

        # determine ground truth ROI's
        idx = dataset.class_label['Car']
        ground_truth_rois = dataset.strip_ROIs(class_ID=idx, label_list=label)
        image_copy = image.copy()
        for g in range(len(ground_truth_rois)):
            p1, p2 = swap_box_xy(ground_truth_rois[g])
            cv2.rectangle(image_copy, p1, p2, color=(0, 255, 0))

        # subdivide image into grid of ROI's
        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
        ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)
        ROI_IoUs = []
        for idx in range(len(ROIs)):
            ROI_IoUs += [anchors.calc_max_IoU(boxes[idx], ground_truth_rois)]

        # create labelled batch of ROI's for current image
        roi_batch = []
        for k in range(len(boxes)):
            p1, p2 = swap_box_xy(boxes[k])
            roi = image[p1[1]:p2[1], p1[0]:p2[0]]
            name_class = 0
            if ROI_IoUs[k] > IoU_threshold:
                name_class = 1
                # cv2.rectangle(image_copy, pt1, pt2, color=(0, 0, 255))
            roi_batch.append((roi, name_class, (p1, p2)))

        # convert batch of ROI's to a data loader
        batch_size = len(roi_batch)
        batch_dataset = ROIBatchDataset(roi_batch, data_transform())
        batch_loader = DataLoader(batch_dataset, batch_size=batch_size, shuffle=False)

        # ----- testing batch of ROI's ----- #
        correct = 0
        predicted = []
        for images, labels, _ in batch_loader:
            images = images.to(device=device)
            labels = labels.to(device=device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            # print(predicted)
            correct += (predicted == labels).sum().item()
            del images, labels, output

        if verbose:
            print('Accuracy of image #{} for {} ROIs: {:.2f} %'.format(i+1, batch_size, 100 * correct / batch_size))

        # determine IoU for predicted car regions
        image_ious = []
        for j in range(len(predicted)):
            if predicted[j] == 1:
                box = roi_batch[j][2]
                cv2.rectangle(image_copy, box[0], box[1], color=(255, 0, 255))
                box = swap_box_xy(box)
                image_ious.append(anchors.calc_max_IoU(box, ground_truth_rois))
                predicted_ious.append(anchors.calc_max_IoU(box, ground_truth_rois))
        if verbose:
            if len(image_ious) == 0:
                print("No cars were detected in image #{}".format(i + 1))
            else:
                print("Mean IoU of image #{} for {} detected cars: {:.4f}".format(i+1, len(image_ious), np.mean(image_ious)))

        # display image with ground truth and predicted bounding boxes
        if show_images:
            cv2.imshow('predicted boxes (pink), ground truth boxes (green)', image_copy)
            key = cv2.waitKey(0)
            if key == ord('x'):
                break

        # increment item num
        i += 1
        if max_data_items > 0 and i >= max_data_items:
            break

    end_time = time.time()

    # ----- print final training statistics ----- #
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    total_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    print("{} test completed...".format(datetime.now()))
    print("elapsed time: {}".format(total_time))
    print("total mean IoU for {} images: {:.4f}".format(len(dataset), np.mean(predicted_ious)))

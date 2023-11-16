import argparse
import time
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)


if __name__ == '__main__':

    max_data_items = 10
    IoU_threshold = 0.02
    device = torch.device('cpu')

    # ----- set up argument parser for command line inputs ----- #
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-i', metavar='input_dir', type=str, help='input dir (./)')
    argParser.add_argument('-m', metavar='model_dir', type=str, help='model dir (./)')
    argParser.add_argument('-d', metavar='display', type=str, help='[y/N]')
    argParser.add_argument('-cuda', metavar='cuda', type=str, help='[y/N]')

    # ----- get args from the arg parser ----- #
    args = argParser.parse_args()
    input_dir = Path(args.i)
    model_dir = Path(args.m)
    show_images = str(args.d).lower() == 'y'
    use_cuda = str(args.cuda).lower() == 'y'

    print("CudaIsAvailable: {}, UseCuda: {}".format(torch.cuda.is_available(), use_cuda))
    if torch.cuda.is_available() and use_cuda:
        print('using cuda ...')
        device = torch.device('cuda')
    else:
        print('using cpu ...')

    # ----- initialize model ----- #
    model = resnet18(weights=None, num_classes=2)
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
    start_time = time.time()
    for item in enumerate(dataset):
        idx = item[0]
        image = item[1][0]
        label = item[1][1]

        # subdivide the image into regions
        idx = dataset.class_label['Car']
        car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)
        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)

        # determine regions that contain a car
        ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)
        ROI_IoUs = []
        for idx in range(len(ROIs)):
            ROI_IoUs += [anchors.calc_max_IoU(boxes[idx], car_ROIs)]

        # create labelled batch of ROI's for current image
        roi_batch = []
        for k in range(len(boxes)):
            name_class = 0
            if ROI_IoUs[k] > IoU_threshold:
                name_class = 1
            box = boxes[k]
            pt1 = (box[0][1], box[0][0])
            pt2 = (box[1][1], box[1][0])
            roi = image[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            roi_batch.append((roi, name_class))

            # if show_images:
            #     print(roi_batch[k][1])
            #     cv2.imshow('roi', roi)
            #     key = cv2.waitKey(0)
            #     if key == ord('x'):
            #         break

        batch_size = len(roi_batch)
        batch_dataset = ROIBatchDataset(roi_batch, data_transform())
        batch_loader = DataLoader(batch_dataset, batch_size=batch_size)

        # ----- testing batch of ROI's ----- #
        correct = 0
        for images, labels in batch_loader:
            images = images.to(device=device)
            labels = labels.to(device=device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()
            del images, labels, output

        print('Accuracy of image #{} for {} ROIs: {:.2f} %'.format(i+1, batch_size, 100 * correct / batch_size))

        # display original image with model's predicted boxes around cars

        # calculate model's IoU score against original image IoU

        i += 1
        print("item #", i)
        if max_data_items > 0 and i >= max_data_items:
            break

    end_time = time.time()

    # ----- print final training statistics ----- #
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    total_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    print("{} test completed...".format(datetime.now()))
    print("elapsed time: {}".format(total_time))
    # print('Accuracy of the network on the {} test images: {:.2f} %'.format(total, 100 * correct / total))

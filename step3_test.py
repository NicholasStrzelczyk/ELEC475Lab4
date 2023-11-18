import argparse
import KittiROIDataset as roids
import time
from datetime import datetime
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import torchsummary
from torchvision.models import resnet18


def data_transform():
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)


def append_confusion_matrix(predicted, labels, conf_matrix):
    tp, tn, fp, fn = conf_matrix
    for i in range(len(predicted)):
        if predicted[i] == labels[i] and labels[i] == 1:
            tp += 1
        elif predicted[i] == labels[i] and labels[i] == 0:
            tn += 1
        elif predicted[i] != labels[i] and labels[i] == 0:
            fp += 1
        else:
            fn += 1
    return tp, tn, fp, fn


if __name__ == '__main__':

    device = torch.device('cpu')

    # ----- set up argument parser for command line inputs ----- #
    parser = argparse.ArgumentParser()
    parser.add_argument('-weight_file', type=str, help='file name for saved model')
    parser.add_argument('-b', type=int, default=20, help='Batch size')
    parser.add_argument('-cuda', type=str, default='n', help='[y/N]')

    # ----- get args from the arg parser ----- #
    args = parser.parse_args()
    batch_size = int(args.b)
    use_cuda = str(args.cuda).lower()
    weight_file = Path(args.weight_file)

    test_set = roids.KittiROIDataset(Path('./../data/Kitti8_ROIs/'), training=False, transform=data_transform())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    test_set_length = len(test_set)

    print("CudaIsAvailable: {}, UseCuda: {}".format(torch.cuda.is_available(), use_cuda))
    if torch.cuda.is_available() and use_cuda == 'y':
        print('using cuda ...')
        device = torch.device('cuda')
    else:
        print('using cpu ...')

    # ----- initialize model ----- #
    model = resnet18(weights=None, num_classes=2)
    model.load_state_dict(torch.load(weight_file))
    model.to(device=device)
    model.eval()
    print('model loaded OK!')

    # ----- begin testing the model ----- #
    torchsummary.summary(model, input_size=(3, 256, 256))
    print("{} testing...".format(datetime.now()))
    start_time = time.time()

    # ----- Testing ----- #
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    conf_matrix = (tp, tn, fp, fn)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device=device)
            labels = labels.to(device=device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            conf_matrix = append_confusion_matrix(predicted, labels, conf_matrix)
            del images, labels, outputs

    end_time = time.time()
    tp, tn, fp, fn = conf_matrix

    # ----- print final training statistics ----- #
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    total_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    print("Total testing time: {}, ".format(total_time))
    print('Accuracy of the network on the {} test images: {:.2f} %'.format(total, 100 * correct / total))
    print('True positives: {}, True negatives: {}'.format(tp, tn))
    print('False positives: {}, False negatives: {}'.format(fp, fn))

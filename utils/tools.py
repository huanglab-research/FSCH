import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from tqdm import tqdm
import torchvision.datasets as dsets
import random
import matplotlib.pyplot as plt


def config_dataset(config):
    if "cifar" in config["dataset"]:
        config["topK"] = 5000
        config["n_class"] = 10
    elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "mirflickr":
        config["topK"] = 5000
        config["n_class"] = 38

    config["data_path"] = "./dataset/" + config["dataset"] + "/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "./data/nuswide_21/NUS-WIDE/"
    if config["dataset"] == "coco":
        config["data_path"] = ""
    config["data"] = {
        "train_set": {"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    return config


draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
              9000, 9500, 10000]

def pr_curve(rF, qF, rL, qL, draw_range=draw_range):
    #  https://blog.csdn.net/HackerTom/article/details/89425729
    n_query = qF.shape[0]
    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    Rank = np.argsort(CalcHammingDist(qF, rF))
    P, R = [], []
    for k in tqdm(draw_range):
        p = np.zeros(n_query)
        r = np.zeros(n_query)
        for it in range(n_query):
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)
            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all
        P.append(np.mean(p))
        R.append(np.mean(r))
    return P, R


class TrainImageList(object):

    def __init__(self, data_path, image_list, train_transform, test_transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.train_transform = train_transform
        self.test_transform = test_transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img1 = self.train_transform(img)
        img2 = self.train_transform(img)
        return img1, img2

    def __len__(self):
        return len(self.imgs)


class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index


class Train_MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2


def cifar_dataset(config):
    batch_size = config["batch_size"]

    train_size = 500
    test_size = 100

    if config["dataset"] == "cifar10-2":
        train_size = 5000
        test_size = 1000

    cifar_dataset_root = './data/cifar10/'
    # Dataset
    train_dataset = Train_MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=train_transform,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=test_transform)

    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=test_transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]
        # print(index.shape)
        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]   #  0-99
            print(test_index.shape)
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False
    # print(test_index.shape)
    
    if config["dataset"] == "cifar10":
        pass
    elif config["dataset"] == "cifar10-1":
        database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2":
        database_index = train_index

    # np.random.seed(0)
    # np.random.shuffle(train_index)
    # np.random.seed(0)
    # np.random.shuffle(test_index)
    # np.random.seed(0)
    # np.random.shuffle(database_index)
    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]


def get_data(config):
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        if data_set == "train_set":
            dsets[data_set] = TrainImageList(config["data_path"],
                                             open(data_config[data_set]["list_path"]).readlines(),
                                             train_transform=train_transform, test_transform=test_transform)
        else:
            dsets[data_set] = ImageList(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=test_transform)
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle=True, num_workers=8)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])


def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        print(img.shape)
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcHammingDist_ch(query, dataset):
    query = query.cpu().numpy().astype(bool)
    dataset = dataset.cpu().numpy().astype.astype(bool)
    hamm = np.bitwise_xor(query, dataset)
    hamm = np.sum(hamm, axis=1)
    return hamm


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

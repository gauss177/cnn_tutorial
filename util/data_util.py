import torch
import torchvision.datasets as dsets
from torch.utils.data import sampler
import torchvision.transforms as transforms
from torchsample.transforms import *


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class MyTransformer(object):
    def __init__(self):
        self.aff = [
            RandomRotate(180),
            RandomTranslate(translation_range=(0.1, 0.1)),
            # RandomShear(10.0),
            RandomZoom([0.8, 1.2]),
            RandomFlip()
        ]
        self.image = [
            RandomBrightness(-0.2, 0.2),
            # RandomGamma(0.5, 1.5),
            # RandomContrast(0.7, 2.0),
            # RandomSaturation(-0.5, 3)
        ]
        self.debug = None
        # self.norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    def _transformer(self, input, p=None):
        # define my transformer
        # output = transforms.Scale(40)(input)
        output = transforms.RandomCrop(32, padding=4)(input)
        output = transforms.ToTensor()(output)
        if p is None:
            p = np.random.random()

        if p < 0.1:
            trans = random.sample(self.aff, 2)
        elif p < 0.5:
            trans = random.sample(self.image, 1)
        elif p < 0.5:
            trans = random.sample(self.aff + self.image, 2)
        else:
            trans = []

        for tr in trans:
            output = tr(output)
        return output

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = self._transformer(_input)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]


def gen_dataset_example(path=None):
    transform = transforms.Compose([
        # transforms.Scale(64),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_data_augmentation = transforms.Compose([
        # transforms.Scale(36),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32),
        transforms.ToTensor(),
        # RandomFlip(),
        RandomRotate(180),
        RandomShear(0.1)
        # RandomTranslate(0.1)
        ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    my_transform = MyTransformer()
    if path is None:
        path = './data/'
    train_dataset = dsets.CIFAR10(root=path,
                                   train=True,
                                   transform=transform,
                                   download=True)

    train_dataset_with_aug = dsets.CIFAR10(root=path,
                                           train=True,
                                           transform=my_transform,
                                           download=True)

    test_dataset = dsets.CIFAR10(root=path,
                                  train=False,
                                  transform=transform)


    # data set setting
    NUM_TRAIN = 49000
    NUM_VAL = 1000
    random_sampler_train = sampler.SubsetRandomSampler(range(0, NUM_TRAIN+NUM_VAL))
    sampler_val = ChunkSampler(NUM_VAL, NUM_TRAIN)

    loader_train = torch.utils.data.DataLoader(train_dataset_with_aug, batch_size=128, sampler=random_sampler_train, num_workers=3)
    loader_val = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=sampler_val, num_workers=1)
    loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=3)
    return loader_train, loader_val, loader_test


def gen_cifar10(transform, path=None):
    my_transform = MyTransformer()
    if path is None:
        path = './data/'
    train_dataset = dsets.CIFAR10(root=path,
                                   train=True,
                                   transform=transform,
                                   download=True)

    train_dataset_with_aug = dsets.CIFAR10(root=path,
                                           train=True,
                                           transform=my_transform,
                                           download=True)

    test_dataset = dsets.CIFAR10(root=path,
                                  train=False,
                                  transform=transform)


    # data set setting
    NUM_TRAIN = 49000
    NUM_VAL = 1000
    random_sampler_train = sampler.SubsetRandomSampler(range(0, NUM_TRAIN+NUM_VAL))
    sampler_val = ChunkSampler(NUM_VAL, NUM_TRAIN)

    loader_train = torch.utils.data.DataLoader(train_dataset_with_aug, batch_size=128, sampler=random_sampler_train, num_workers=3)
    loader_val = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=sampler_val, num_workers=1)
    loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=3)
    return loader_train, loader_val, loader_test
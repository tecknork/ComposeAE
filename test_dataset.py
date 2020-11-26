import datasets
import torchvision

if __name__ == '__main__':
    print("hello World")
    trainset = datasets.MITStates(
        path='data/mitstates',
        split='test',
        data_type='GCZSL',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))

    print(len(trainset))

    # testset = datasets.MITStates(
    #     path='data/mitstates',
    #     split='test',
    #     transform=torchvision.transforms.Compose([
    #         torchvision.transforms.Resize(224),
    #         torchvision.transforms.CenterCrop(224),
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize([0.485, 0.456, 0.406],
    #                                          [0.229, 0.224, 0.225])
    #     ]))

    # print(len(testset))
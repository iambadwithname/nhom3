# uhh why put code in this file
# cause torch worker is dumb(cant find the data) thats why
import kagglehub
import torchvision
from torchvision import datasets, transforms

##importing data
datapath = kagglehub.dataset_download(
    "sachinkumar413/covid-pneumonia-normal-chest-xray-images"
)
# [ "0.5067335367202759", "0.23598085343837738" ]
normalize = transforms.Normalize(
    mean=0.5067335367202759,
    std=0.23598085343837738,
)
normalizei = transforms.Normalize(
    mean=-normalize.mean / normalize.std,
    std=1 / normalize.std,
)
transformi = transforms.Compose(
    [
        normalizei,
        torchvision.transforms.ToPILImage(),
    ]
)


def remove_dim(x):
    return x[:1]


transform = transforms.Compose(
    [
        transforms.CenterCrop((232, 232)),
        transforms.ToTensor(),
        transforms.Lambda(remove_dim),
        normalize,
    ]
)
dataset = datasets.ImageFolder(datapath, transform=transform)

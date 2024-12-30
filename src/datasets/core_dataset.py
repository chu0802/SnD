from src.datasets.base import BaseClassificationDataset, BaseUnlabeledDataset


class ImageNet(BaseClassificationDataset):
    dataset_name = "imagenet"
    annotation_filename = "annotations_new.json"


class Caltech101(BaseClassificationDataset):
    dataset_name = "caltech-101"
    annotation_filename = "split_zhou_Caltech101_new.json"


class OxfordPets(BaseClassificationDataset):
    dataset_name = "oxford-pets"
    annotation_filename = "split_zhou_OxfordPets_new.json"


class StanfordCars(BaseClassificationDataset):
    dataset_name = "stanford-cars"
    annotation_filename = "split_zhou_StanfordCars_new.json"


class Flowers102(BaseClassificationDataset):
    dataset_name = "flowers-102"
    annotation_filename = "split_zhou_OxfordFlowers_new.json"


class Food101(BaseClassificationDataset):
    dataset_name = "food-101"
    annotation_filename = "split_zhou_Food101_new.json"


class FGVCAircraft(BaseClassificationDataset):
    dataset_name = "fgvc-aircraft"
    annotation_filename = "annotations_new.json"


class EuroSAT(BaseClassificationDataset):
    dataset_name = "eurosat"
    annotation_filename = "split_zhou_EuroSAT_new.json"


class UCF101(BaseClassificationDataset):
    dataset_name = "ucf-101"
    annotation_filename = "split_zhou_UCF101_new.json"


class DTD(BaseClassificationDataset):
    dataset_name = "dtd"
    annotation_filename = "split_zhou_DescribableTextures_new.json"


if __name__ == "__main__":
    dataset = DTD(root="/work/chu980802/data/classification", mode="train")
    print(dataset[0])
    print(len(dataset))
    print(len(dataset.class_name_list))

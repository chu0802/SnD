from src.datasets.base import BaseClassificationDataset, BaseUnlabeledDataset


class ImageNet(BaseClassificationDataset):
    dataset_name = "imagenet"
    annotation_filename = "imagenet_annotations.json"


class Caltech101(BaseClassificationDataset):
    dataset_name = "caltech-101"
    annotation_filename = "caltech101_annotations.json"


class OxfordPets(BaseClassificationDataset):
    dataset_name = "oxford-pets"
    annotation_filename = "oxfordpets_annotations.json"


class StanfordCars(BaseClassificationDataset):
    dataset_name = "stanford-cars"
    annotation_filename = "stanfordcars_annotations.json"


class Flowers102(BaseClassificationDataset):
    dataset_name = "flowers-102"
    annotation_filename = "flowers102_annotations.json"


class Food101(BaseClassificationDataset):
    dataset_name = "food-101"
    annotation_filename = "food101_annotations.json"


class FGVCAircraft(BaseClassificationDataset):
    dataset_name = "fgvc-aircraft"
    annotation_filename = "aircraft_annotations.json"


class EuroSAT(BaseClassificationDataset):
    dataset_name = "eurosat"
    annotation_filename = "eurosat_annotations.json"


class UCF101(BaseClassificationDataset):
    dataset_name = "ucf-101"
    annotation_filename = "ucf101_annotations.json"


class DTD(BaseClassificationDataset):
    dataset_name = "dtd"
    annotation_filename = "dtd_annotations.json"


if __name__ == "__main__":
    dataset = DTD(root="/work/chu980802/data/classification", mode="train")
    print(dataset[0])
    print(len(dataset))
    print(len(dataset.class_name_list))

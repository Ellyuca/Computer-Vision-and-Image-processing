from PIL import Image
from os import walk
from os import path
import numpy as np
from random import shuffle
import json


def main():
    data = []
    images = {}
    report = {
        'classes': {}
    }

    ##
    # Don't needed. DataSet Class do it for us...
    # normalize_data = np.vectorize(lambda elm: elm / 255.)

    for root, dirs, files in walk("coil-20-proc"):
        for file_ in files:
            class_ = int(file_.split("__")[0].replace("obj", "")) - 1
            if class_ not in report['classes']:
                report['classes'][class_] = path.join(root, file_)
            print("+ Load image {}".format(file_), end="\r")
            if class_ not in images:
                images[class_] = []
            images[class_].append(path.join(root, file_))

    for key in images:
        shuffle(images[key])

    train_set = []
    test_set = []
    samples_x_class = 12

    print("+ Select test images and train images...")

    for class_, list_ in images.items():
        to_test = list_[:samples_x_class]
        to_train = list_[samples_x_class:]

        one_hot = np.full([20, ], 0.)
        one_hot[class_] = 1

        for file_path in to_test:
            image = Image.open(file_path)
            # image = image.resize((28, 28), Image.ANTIALIAS)
            test_set.append([
                one_hot,
                # normalize_data(
                np.array(image).ravel()
                # )
            ])
        for file_path in to_train:
            image = Image.open(file_path)
            # image = image.resize((28, 28), Image.ANTIALIAS)
            train_set.append([
                one_hot,
                # normalize_data(
                np.array(image).ravel()
                # )
            ])


    print("+ Imported all images")

    # print(data[0][1].shape)
    # print(max(data[0][1]))

    print("+ Get train")
    train_labels = np.array([elm[0]
                             for elm in train_set], dtype=np.uint8)
    train_data = np.array([elm[1]
                           for elm in train_set], dtype=np.float32)
    print("+ Get test set")
    test_labels = np.array([elm[0]
                            for elm in test_set], dtype=np.uint8)
    test_data = np.array([elm[1]
                          for elm in test_set], dtype=np.float32)

    report['train_data_shape'] = train_data.shape
    report['test_data_shape'] = test_data.shape
    report['train_labels_shape'] = train_labels.shape
    report['test_labels_shape'] = test_labels.shape

    files_to_store = [
        ('coil_train_labels.bin', train_labels),
        ('coil_train_data.bin', train_data),
        ('coil_test_labels.bin', test_labels),
        ('coil_test_data.bin', test_data)
    ]

    print("+ Write dataset")
    for file_name, data in files_to_store:
        with open(file_name, 'wb') as cur_file:
            cur_file.write(data.tobytes())

    print("+ Write report")
    with open("coil_dataset_report.json", "w") as report_file:
        json.dump(report, report_file, indent=2)


if __name__ == '__main__':
    main()

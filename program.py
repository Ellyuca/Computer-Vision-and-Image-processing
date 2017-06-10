#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
# TF
import tensorflow as tf
import numpy as np
from coil_conv import gen_model
from PIL import Image
from PIL.ImageQt import ImageQt
import json
import matplotlib.pyplot as plt
import io


COIL_CLASSE_NAMES = [
    "duck",
    "arrow",
    "car_1",
    "cat",
    "anacin",
    "car_2",
    "structure",
    "powder",
    "tylenol",
    "vaseline",
    "mushroom",
    "mug",
    "pig",
    "obj",
    "cylinder",
    "container",
    "jar",
    "cup",
    "car_3",
    "philadelphia"
]


class MyProgram(QWidget):

    def __init__(self):
        super(MyProgram, self).__init__()

        self.initTF()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Coil Tester')

        self.input_label = QLabel()
        self._drop_pixmap = QPixmap('prog_data/drop.png')
        self._drop_pixmap.scaled(128, 128)
        self.input_label.setPixmap(self._drop_pixmap)

        arrow = QLabel()
        arrow.setPixmap(QPixmap('prog_data/arrow.png'))

        self.output_label = QLabel()
        self._result = QPixmap('prog_data/result.png')
        self._result.scaled(128, 128)
        self.output_label.setPixmap(self._result)

        self.mpl_canvas = QLabel()

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.setAlignment(Qt.AlignCenter)
        hbox.addWidget(self.input_label)
        hbox.addWidget(arrow)
        hbox.addWidget(self.output_label)

        vbox.addLayout(hbox)
        vbox.addWidget(self.mpl_canvas)
        vbox.setAlignment(Qt.AlignCenter)

        self.setLayout(vbox)

        self.setAcceptDrops(True)

        self.show()

    def initTF(self):
        with open("coil_dataset_report.json", "r") as report_file:
            self.report = json.load(report_file)

        self.model = gen_model()

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(self.model.variables)
        saver.restore(self.sess, "models/coil.chk")

    def dropEvent(self, evt):
        # print(evt.mimeData())
        # print(evt.mimeData().urls()[0].toLocalFile())

        file_name = evt.mimeData().urls()[0].toLocalFile()

        pixmap = QPixmap(file_name)
        self.input_label.setPixmap(pixmap)

        img = np.array(Image.open(file_name), dtype=np.float32).ravel()
        img = img / 255.

        res = self.sess.run(self.model.y_conv, feed_dict={
            self.model.x: [img],
            self.model.keep_prob: 1.0
        })[0]

        # print(res, np.argmax(res))

        idx = str(np.argmax(res))

        pixmap = QPixmap(self.report['classes'][idx])
        self.output_label.setPixmap(pixmap)

        min_ = np.min(res)
        max_ = np.max(res)
        # print(max_, min_)

        # normalized = [
        #     ((elm - min_) / (max_ - min_)) * 100. for elm in res
        # ]

        # print(normalized)

        fig = plt.figure(1)
        positions = np.arange(
            len(res  # normalized
                ))
        plt.barh(
            positions,
            # normalized
            res
        )
        plt.yticks(positions, COIL_CLASSE_NAMES)
        plt.xticks([min_, 0, max_], [str(min_), "0", str(max_)])
        plt.xlabel('Percentage')

        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf,
                    format='png',
                    pad_inches=0.42,
                    transparent=True,
                    bbox_inches="tight"
                    )
        buf.seek(0)
        plt.close(fig)

        img = ImageQt(Image.open(buf))
        pixmap = QPixmap.fromImage(img)
        self.mpl_canvas.setPixmap(pixmap)

    def dragEnterEvent(self, evt):
        if evt.mimeData().hasUrls():
            evt.accept()
        else:
            self.input_label.setPixmap(self._drop_pixmap)
            evt.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_program = MyProgram()
    sys.exit(app.exec_())

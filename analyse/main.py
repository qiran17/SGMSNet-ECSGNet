import sys
import os
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow,QAction,QMessageBox,QFileDialog
from PyQt5.QtGui import QPixmap,QTextCharFormat, QFont, QColor
from classify_ui import Ui_MainWindow
import torch
import torchvision.transforms as transforms
from get_model import model_list,get_model

class Main_Window(QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super(Main_Window, self).__init__(parent)
        self.setupUi(self)

        # 选择模型
        self.comboBox.addItems(model_list)
        self.textEdit.append('你已选择' + self.comboBox.currentText() + '模型。')
        # self.model_name=self.comboBox.currentText()
        self.model = self.load_model(self.comboBox.currentText())
        self.comboBox.currentTextChanged.connect(self.update_model_mess)
        self.pushButton.setEnabled(False)
        # 加载图片文件夹路径
        self.toolButton.clicked.connect(self.get_image_path)

        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        # 上一张
        self.pushButton_2.clicked.connect(self.show_prev_image)

        # 下一张
        self.pushButton_3.clicked.connect(self.show_next_image)

        self.pushButton.clicked.connect(self.predict_image)

    def update_model_mess(self):
        self.textEdit.append('你已选择' + self.comboBox.currentText() + '模型。')
        self.model = self.load_model(self.comboBox.currentText())


    def get_image_path(self):
        self.images_path = QFileDialog.getExistingDirectory(self, 'Select Folder', './')

        if self.images_path:
            self.pushButton.setEnabled(True)
            self.lineEdit.setText(self.images_path)
            self.image_files = [filename for filename in os.listdir(self.images_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            self.current_image_index = 0
            self.show_image()


    def show_image(self):
        if self.image_files:
            self.image_path = os.path.join(self.images_path, self.image_files[self.current_image_index])
            try:
                pixmap = QPixmap(self.image_path)
                self.graphics_scene.clear()
                self.graphics_scene.addPixmap(pixmap.scaled(380, 380))
            except Exception as e:
                self.textEdit.append('加载错误，请检查图片格式')
            if len(self.image_files)==1:
                self.pushButton_2.setEnabled(False)
                self.pushButton_3.setEnabled(False)
            elif self.current_image_index == 0:
                self.pushButton_2.setEnabled(False)
                self.pushButton_3.setEnabled(True)

            elif self.current_image_index == len(self.image_files) - 1:
                self.pushButton_2.setEnabled(True)
                self.pushButton_3.setEnabled(False)
            else:
                self.pushButton_2.setEnabled(True)
                self.pushButton_3.setEnabled(True)

        else:
            self.textEdit.append('该文件夹内没有符合要求的图片，请重新选择文件夹！')

            self.pushButton.setEnabled(False)

    def show_prev_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image()

    def show_next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_image()

    def load_model(self, model_name):
        model_path,self.class_name = get_model(model_name)
        # 加载PyTorch模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path)
        model.to(device)
        model.eval()
        return model

    def preprocess_image(self, image_path):
        # 对图像进行预处理
        image = Image.open(image_path)
        image = image.convert('RGB')
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.44127703, 0.4712498, 0.43714803], std=[0.18507297, 0.18050247, 0.16784933])

        ])
        preprocessed_image = transform(image)
        return preprocessed_image.unsqueeze(0)  # 添加批处理维度

    def predict_image(self):
        if self.image_path:
            # 对选择的图片进行预测
            try:
                processed_image = self.preprocess_image(self.image_path)
            except Exception as e:
                self.textEdit.append('加载图片失败，请检查图片格式')
            with torch.no_grad():
                processed_image = processed_image.cuda()
                _, outputs = self.model(processed_image, processed_image)
            # 在这里处理模型输出并显示预测结果
            if self.class_name:

                cursor = self.textEdit.textCursor()
                self.original_format = cursor.charFormat()
                cursor.insertBlock()
                cursor.insertText("这张图片是 ")
                # 创建要插入的文本格式
                format_bold_red = QTextCharFormat()
                format_bold_red.setFontWeight(QFont.Bold)
                format_bold_red.setForeground(QColor('red'))
                format_bold_red.setFontPointSize(12)  # 设置字号
                format_bold_red.setFontFamily("Arial")  # 设置字体

                # 将格式应用到文本
                cursor.setCharFormat(format_bold_red)

                # 在光标处插入文本
                cursor.insertText(str(self.class_name[outputs.argmax().item()]))

                # 恢复默认格式（可选）
                cursor.setCharFormat(self.original_format)
                # 换行
                cursor.insertBlock()
                # 滚动至最下方显示最后一行文本
                scrollbar = self.textEdit.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
            else:
                cursor = self.textEdit.textCursor()
                self.original_format = cursor.charFormat()
                cursor.insertBlock()
                cursor.insertText("这张图片是类别 ")
                # 创建要插入的文本格式
                format_bold_red = QTextCharFormat()
                format_bold_red.setFontWeight(QFont.Bold)
                format_bold_red.setForeground(QColor('red'))
                format_bold_red.setFontPointSize(12)  # 设置字号
                format_bold_red.setFontFamily("Arial")  # 设置字体

                # 将格式应用到文本
                cursor.setCharFormat(format_bold_red)

                # 在光标处插入文本
                cursor.insertText(str(outputs.argmax().item()))

                # 恢复默认格式（可选）
                cursor.setCharFormat(self.original_format)
                # 换行
                cursor.insertBlock()
                # 滚动至最下方显示最后一行文本
                scrollbar = self.textEdit.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())


            


if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建应用程序对象
    mainwindow = Main_Window()  # 创建主窗口
    mainwindow.show()  # 显示主窗口
    sys.exit(app.exec_())  # 在主线程中退出

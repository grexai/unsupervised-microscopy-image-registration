import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QVBoxLayout, QWidget, QPushButton, QLabel, QHBoxLayout, QSizeGrip,QMessageBox
)
from PyQt5.QtWidgets import QFileDialog, QDialog
from PyQt5.QtWidgets import QListWidget, QListWidgetItem


class ImageSelectDialog(QDialog):
    def __init__(self):
        super(ImageSelectDialog, self).__init__()

        self.selected_images = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Add a button to select the first image
        button_1 = QPushButton("Select Image 1")
        button_1.clicked.connect(self.select_image_1)
        layout.addWidget(button_1)

        # Add a button to select the second image
        button_2 = QPushButton("Select Image 2")
        button_2.clicked.connect(self.select_image_2)
        layout.addWidget(button_2)

        # Add a button to confirm the selection
        confirm_button = QPushButton("OK")
        confirm_button.clicked.connect(self.accept)
        layout.addWidget(confirm_button)

        self.setLayout(layout)

    def select_image_1(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image 1", "", "Images (*.png *.jpg *.bmp)")
        if image_path:
            self.selected_images.append(image_path)

    def select_image_2(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image 2", "", "Images (*.png *.jpg *.bmp)")
        if image_path:
            self.selected_images.append(image_path)

    def get_selected_images(self):
        return tuple(self.selected_images)


class LandmarkPainter(QWidget):
    def __init__(self):
        super(LandmarkPainter, self).__init__()
        self.overlay_window = None  # Store the OverlayWindow instance
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.image_1 = None
        self.image_2 = None
        self.image_item_1 = QGraphicsPixmapItem()
        self.image_item_2 = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item_1)
        self.scene.addItem(self.image_item_2)
        self.image1_dims = ()  # Tuple to store dimensions of image1
        self.image2_dims = ()  # Tuple to store dimensions of image2

        self.landmarks = {'image1': [], 'image2': []}  # Dictionary to store landmarks for both images

        # Set up the UI
        self.setup_ui()

    def setup_ui(self):
        # Create a QVBoxLayout for the main widget
        main_layout = QHBoxLayout(self)

        # Create a QVBoxLayout for the image view and buttons
        image_layout = QVBoxLayout()

        button_select_images = QPushButton("Open Images")
        button_select_images.clicked.connect(self.show_image_select_dialog)

        button_save_landmarks = QPushButton("Save Landmarks")
        button_save_landmarks.clicked.connect(self.save_landmarks)

        button_overlay_images = QPushButton("Overlay Images")
        button_overlay_images.clicked.connect(self.show_overlay_window)

        # Add buttons to the image layout
        image_layout.addWidget(self.view)
        image_layout.addWidget(button_select_images)
        image_layout.addWidget(button_save_landmarks)
        image_layout.addWidget(button_overlay_images)

        # Create buttons for point operations
        button_remove_point = QPushButton("Remove Selected Point")
        button_remove_point.clicked.connect(self.remove_selected_point)

        # Add the point operation buttons to the image layout
        image_layout.addWidget(button_remove_point)
        # Add the image layout to the main layout
        main_layout.addLayout(image_layout)

        # Create a QVBoxLayout for the list widget
        list_layout = QVBoxLayout()

        # Add a label to the list layout
        list_layout.addWidget(QLabel("Landmark Points:"))

        # Add a QListWidget to the list layout
        self.landmark_list_widget = QListWidget()

        self.landmark_list_widget.setFixedWidth(200)
        list_layout.addWidget(self.landmark_list_widget)

        # Add the list layout to the main layout
        main_layout.addLayout(list_layout)
        # Set the layout for the main window
        self.setWindowTitle("Landmark Painter")
        self.setFixedSize(1480, 720)

    def remove_selected_point(self):
        selected_items = self.landmark_list_widget.selectedItems()

        for item in selected_items:
            index = self.landmark_list_widget.row(item)
            image_index, point_index = self.get_point_info_from_list_index(index)


            # Remove the point from the landmarks dictionary
            if image_index in self.landmarks and point_index < len(self.landmarks[image_index]):
                del self.landmarks[image_index][point_index]

            # Remove the item from the list widget
            self.landmark_list_widget.takeItem(index)


        # Update the scene with the modified landmarks
        self.update_scene()

    def get_point_info_from_list_index(self, index):
        # Extract image and point indices from the list index
        if index < len(self.landmarks['image1']):
            return 'image1', index
        else:
            return 'image2', index - len(self.landmarks['image1'])

    def update_scene(self):
        # Clear the scene
        # self.scene.clear()

        # Create new QGraphicsPixmapItems with the original images
        # if self.image_item_1.pixmap() is not None:
        #     image_item_1_copy = QGraphicsPixmapItem(self.image_item_1.pixmap())
        #     self.scene.addItem(image_item_1_copy)
        #
        # if self.image_item_2.pixmap() is not None:
        #     image_item_2_copy = QGraphicsPixmapItem(self.image_item_2.pixmap())
        #     self.scene.addItem(image_item_2_copy)

        # Draw the existing landmarks
        for image_index, points in self.landmarks.items():

            for point in points:
                if image_index == 2:
                    x = point['x'] + self.image1_dims[0]
                else:
                    x = point['x']
                self.draw_landmark(QPointF(x, point['y']),redraw=True)

    def show_image_select_dialog(self):
        # Create an instance of the ImageSelectDialog
        image_select_dialog = ImageSelectDialog()

        # Check if the user pressed OK
        if image_select_dialog.exec_() == QDialog.Accepted:
            # Get the selected image paths
            image_path_1, image_path_2 = image_select_dialog.get_selected_images()

            # Load the selected images
            self.load_images(image_path_1, image_path_2)

    def load_images(self, image_path_1, image_path_2):

        # Load the images
        self.image_1 = QImage(image_path_1)
        self.image_2 = QImage(image_path_2)

        # Set the dimensions of image1 and image2
        self.image1_dims = (self.image_1.width(), self.image_1.height())
        self.image2_dims = (self.image_2.width(), self.image_2.height())

        # Create a combined image
        combined_image = QImage(self.image1_dims[0] + self.image2_dims[0],
                                max(self.image1_dims[1], self.image2_dims[1]), QImage.Format_RGB32)
        combined_image.fill(Qt.white)

        # Paint the images onto the combined image
        painter = QPainter(combined_image)
        painter.drawImage(0, 0, self.image_1)
        painter.drawImage(self.image1_dims[0], 0, self.image_2)
        painter.end()

        # Set the combined image
        self.image_item_1.setPixmap(QPixmap.fromImage(combined_image))

    def save_landmarks(self):
        # Replace this path with your desired path for saving landmarks
        save_path = "./landmarks.txt"

        # Combine landmarks for both images into a single dictionary
        combined_landmarks = {'image1': self.landmarks['image1'], 'image2': self.landmarks['image2']}
        # Save the combined landmarks to a text file
        with open(save_path, 'w') as file:
            file.write(str(combined_landmarks))

        print(f"Landmarks saved to: {save_path}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Map the mouse position to scene coordinates
            pos_in_scene = self.view.mapToScene(event.pos())

            # Draw a point on the image and store the landmark
            self.draw_landmark(pos_in_scene)

    def draw_landmark(self, position, redraw=False):
        if len(self.image1_dims) > 0  and len(self.image2_dims > 0):
            # Set the color and size for the landmarks
            color = QColor(255, 0, 0)  # Red
            radius = 5
            # Convert mouse position to scene coordinates

            # Get the width of the first image
            image_1_width = self.image1_dims[0]

            # Determine which image the landmark belongs to
            if position.x() < image_1_width:
                # Landmark is in the first image
                image_index = 1
                landmark_item = self.scene.addEllipse(
                    position.x(),
                    position.y(),
                    radius,
                    radius,
                    QPen(color),
                    QBrush(color)
                )
                if not redraw:
                    self.landmarks['image1'].append({'x': position.x(), 'y': position.y()})
            else:
                # Landmark is in the second image
                image_index = 2
                landmark_item = self.scene.addEllipse(
                    position.x(),
                    position.y(),
                    radius,
                    radius,
                    QPen(color),
                    QBrush(color)
                )
                if not redraw:
                    self.landmarks['image2'].append({'x': position.x() - image_1_width, 'y': position.y()})

            # Store the landmark position
            self.scene.addItem(landmark_item)

            # Add a label indicating the index of the landmark with a small box
            label_rect = self.scene.addRect(position.x() - radius / 2, position.y() - radius / 2 - 10, 30, 20, QPen(),
                                            QBrush(Qt.white))
            label = self.scene.addText(f"{image_index}.{len(self.landmarks[f'image{image_index}'])}")
            label.setPos(position.x() - radius / 2 + 3, position.y() - radius / 2 - 11)
            self.update_landmark_list()
        else:
            QMessageBox.critical(self, f"No images opened",f"Please load image 1 and image2", QMessageBox.Ok,
                                 QMessageBox.Ok)

    def update_landmark_list(self):
        # Clear the existing items in the list widget
        self.landmark_list_widget.clear()

        # Add landmark points for image 1
        for i, point in enumerate(self.landmarks['image1']):
            item = QListWidgetItem(f"Image 1 - Point {i + 1}: ({point['x']}, {point['y']})")
            self.landmark_list_widget.addItem(item)

        # Add landmark points for image 2
        for i, point in enumerate(self.landmarks['image2']):
            item = QListWidgetItem(f"Image 2 - Point {i + 1}: ({point['x']}, {point['y']})")
            self.landmark_list_widget.addItem(item)

    def show_overlay_window(self):
        # Create an OverlayWindow instance if it doesn't exist
        if not self.overlay_window:
            self.overlay_window = OverlayWindow()

        # Use the existing instance
        self.overlay_window.set_images_with_landmarks(self.image_1, self.image_2, self.landmarks)
        self.overlay_window.show()


class OverlayWindow(QWidget):
    def __init__(self, parent=None):
        super(OverlayWindow, self).__init__(parent)
        self.image_label = QLabel()
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        self.setWindowTitle("Image Overlay")
        # Add a QSizeGrip for window resizing
        size_grip = QSizeGrip(self)
        self.setFixedSize(1280, 720)

    def set_images_with_landmarks(self, image_1, image_2, landmarks_data, alpha=0.5):
        if image_1 and image_2:
            landmarks_image1 = landmarks_data.get('image1', [])
            landmarks_image2 = landmarks_data.get('image2', [])
            if len(landmarks_image1) > 3 and len(landmarks_image2) > 3:
                # Convert QImages to NumPy arrays
                array_1 = self.qimage_to_numpy(image_1)
                array_2 = self.qimage_to_numpy(image_2)

                # Ensure arrays are in RGB format
                array_1_rgb = cv2.cvtColor(array_1, cv2.COLOR_BGR2RGB)
                array_2_rgb = cv2.cvtColor(array_2, cv2.COLOR_BGR2RGB)

                # Crop the second image to the size of the first image
                array_2_cropped = array_2_rgb[:array_1_rgb.shape[0], :array_1_rgb.shape[1]]

                # Ensure the cropped image has an alpha channel
                if image_2.hasAlphaChannel():
                    array_2_cropped = cv2.cvtColor(array_2_cropped, cv2.COLOR_RGB2RGBA)

                # Resize the second image to match the dimensions of the first image
                array_2_resized = cv2.resize(array_2_cropped, (array_1_rgb.shape[1], array_1_rgb.shape[0]))

                # Convert dictionaries to np.float32 array
                landmarks_image1_array = np.array([[point['x'], point['y']] for point in landmarks_image1],
                                                  dtype=np.float32)
                landmarks_image2_array = np.array([[point['x'], point['y']] for point in landmarks_image2],
                                                  dtype=np.float32)
                min_len = min(len(landmarks_image1_array), len(landmarks_image2_array))

                # Estimate affine transformation using the smaller array
                M, _ = cv2.estimateAffine2D(landmarks_image2_array[:min_len], landmarks_image1_array[:min_len],
                                            method=cv2.LMEDS)

                # Apply the transformation to image_2
                rows, cols, _ = array_2_resized.shape
                array_2_transformed = cv2.warpAffine(array_2_resized, M, (cols, rows))

                print(f"array_1_rgb.shape: {array_1_rgb.shape}")
                print(f"array_2_transformed.shape: {array_2_transformed.shape}")

                # Remove alpha channel if present in array_2_transformed
                array_2_transformed_rgb = array_2_transformed[:, :, :3]

                # Blend the images using alpha
                result_array = cv2.addWeighted(array_1_rgb, 1 - alpha, array_2_transformed_rgb, alpha, 0)
                print("result_array.shape:", result_array.shape)
                # Create a QPixmap from the combined NumPy array
                # combined_pixmap = QPixmap.fromImage(
                #     QImage(result_array.data, result_array.shape[1], result_array.shape[0],
                #            result_array.shape[1] * result_array.shape[2],
                #            QImage.Format_RGB888 if not image_2.hasAlphaChannel() else QImage.Format_RGBA8888)
                # )
                combined_pixmap = QPixmap.fromImage(
                    QImage(result_array.data, result_array.shape[1], result_array.shape[0],
                           result_array.shape[1] * result_array.shape[2], QImage.Format_RGB888)
                )

                # Set the image to the QLabel widget
                self.image_label.setPixmap(combined_pixmap)

    def qimage_to_numpy(self, qimage):
        width = qimage.width()
        height = qimage.height()

        # Get the image data
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # 4 channels (RGBA)

        return arr


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LandmarkPainter()
    window.show()
    sys.exit(app.exec_())

import sys
import numpy as np

from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QWidget, \
    QPushButton, QLabel, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush, QTransform
from PyQt5.QtCore import Qt, QBuffer, QIODevice
import numpy as np
from PIL import Image
import io
import cv2
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QByteArray


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
        # Create a layout for the main window
        layout = QVBoxLayout(self)

        # Load the image pair
        self.load_image_pair()

        # Create a button to save landmarks
        save_button = QPushButton("Save Landmarks")
        save_button.clicked.connect(self.save_landmarks)
        # Create a button to open the overlay window
        overlay_button = QPushButton("Overlay Images")
        overlay_button.clicked.connect(self.show_overlay_window)

        layout.addWidget(self.view)
        layout.addWidget(save_button)
        layout.addWidget(overlay_button)

        # Set the layout for the main window
        self.setWindowTitle("Landmark Painter")

    def load_image_pair(self):
        # Replace these paths with your image pair paths

        image_path_2 = "d:/datasets/Image_registration/211109-HK-60x/registration/p1_wA1_t1_m1_c1_z0_l1_o0.png"
        image_path_1 = "d:/datasets/Image_registration/211109-HK-60x/LMD63x/p1_wA1_t1_m1_c1_z0_l1_o0_1.BMP"

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

    def draw_landmark(self, position):
        # Set the color and size for the landmarks
        color = QColor(255, 0, 0)  # Red
        radius = 3

        # Get the width of the first image
        image_1_width = self.image1_dims[0]

        # Determine which image the landmark belongs to
        if position.x() < image_1_width:
            # Landmark is in the first image
            image_index = 1
            landmark_item = self.scene.addEllipse(
                position.x() - radius / 2,
                position.y() - radius / 2,
                radius,
                radius,
                QPen(color),
                QBrush(color)
            )
            self.landmarks['image1'].append({'x': position.x(), 'y': position.y()})
        else:
            # Landmark is in the second image
            image_index = 2
            landmark_item = self.scene.addEllipse(
                position.x() - radius / 2,
                position.y() - radius / 2,
                radius,
                radius,
                QPen(color),
                QBrush(color)
            )
            self.landmarks['image2'].append({'x': position.x() - image_1_width, 'y': position.y()})

        # Store the landmark position
        self.scene.addItem(landmark_item)

        # Add a label indicating the index of the landmark with a small box
        label_rect = self.scene.addRect(position.x() - radius / 2, position.y() - radius / 2 - 10, 30, 20, QPen(),
                                        QBrush(Qt.white))
        label = self.scene.addText(f"{image_index}.{len(self.landmarks[f'image{image_index}'])}")
        label.setPos(position.x() - radius / 2 + 3, position.y() - radius / 2 - 11)

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

    def set_images_with_landmarks(self, image_1, image_2, landmarks_data, alpha=0.5):
        if image_1 and image_2 and landmarks_data:
            landmarks_image1 = landmarks_data.get('image1', [])
            landmarks_image2 = landmarks_data.get('image2', [])

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


def main():
    app = QApplication(sys.argv)
    window = LandmarkPainter()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

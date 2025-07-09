import numpy as np
import os
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

img_cols, img_rows = 64, 64

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image):
    img_np = np.array(image)
    faces = face_cascade.detectMultiScale(img_np, scaleFactor=1.1, minNeighbors=5)
    return faces

def get_head_center(face):
    x, y, w, h = face
    center_x = x + w / 2
    center_y = y + h / 2
    return center_x, center_y

def load_images_head_motion_contact(directory, label_name=""):
    images = []
    head_motion_logs = []
    contact_logs = []

    prev_head_center = None
    prev_img_np = None

    total_files = 0
    loaded_files = 0

    for root, dirs, files in os.walk(directory):
        files = sorted(files)
        for file in files:
            total_files += 1
            image_path = os.path.join(root, file)
            try:
                img = Image.open(image_path).convert('L').resize((img_cols, img_rows))
                img_np = np.array(img).astype(np.float32) / 255.0

                # --- Head motion detection ---
                faces = detect_face(img)
                if len(faces) > 0:
                    head_center = get_head_center(faces[0])
                    if prev_head_center is not None:
                        dx = (head_center[0] - prev_head_center[0]) / img_cols
                        dy = (head_center[1] - prev_head_center[1]) / img_rows
                        motion_dist = np.sqrt(dx**2 + dy**2)
                        if motion_dist > 0.01:
                            print(f"[{label_name}] Head motion in {file} | Δ: {motion_dist:.4f}")
                        head_motion_logs.append(motion_dist)
                    prev_head_center = head_center
                else:
                    print(f"No face detected in {file}")
                    prev_head_center = None

                # --- Contact detection (frame difference) ---
                if prev_img_np is not None:
                    diff_img = np.abs(img_np - prev_img_np)
                    motion_pixels = np.sum(diff_img > 0.2)
                    if motion_pixels > 50:
                        print(f"[{label_name}] Contact detected in {file} | Changed Pixels: {motion_pixels}")
                        contact_logs.append((file, motion_pixels))
                prev_img_np = img_np

                images.append(img_np.flatten())
                loaded_files += 1

            except Exception as e:
                print(f"Skipping file {file} due to error: {e}")
                continue

    print(f"[{label_name}] Loaded {loaded_files} images out of {total_files} files.")
    print(f"[{label_name}] Total head motions detected: {len(head_motion_logs)}")
    print(f"[{label_name}] Total contacts detected: {len(contact_logs)}")
    return images

def main():
    abnormal_images = load_images_head_motion_contact("C:/Users/admin/Desktop/tryproject/abnormall", label_name="Abnormal")
    abnormal_labels = np.ones(len(abnormal_images))

    normal_images = load_images_head_motion_contact("C:/Users/admin/Desktop/tryproject/normal", label_name="Normal")
    normal_labels = np.zeros(len(normal_images))

    images = np.vstack((abnormal_images, normal_images))
    labels = np.hstack((abnormal_labels, normal_labels))
    images, labels = shuffle(images, labels, random_state=2)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=1)

    X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 1).astype('float32')

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    model = Sequential([
        Conv2D(64, kernel_size=3, activation='relu', input_shape=(img_cols, img_rows, 1)),
        Conv2D(32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)

    _, accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", accuracy)

    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=["Normal", "Abnormal"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_classes))

    model.save("modellll.h5")
    print("Model saved as modellll.h5")

if __name__ == "__main__":
    main()




























# def main():
#     import numpy as np
#     import os
#     from PIL import Image
#     from sklearn.model_selection import train_test_split
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Conv2D, Flatten, Dense
#     from tensorflow.keras.utils import to_categorical
#     from sklearn.utils import shuffle
#     from sklearn.metrics import classification_report, confusion_matrix


#     img_cols, img_rows = 64, 64

#     # Function to load images from a directory
#     def load_images_from_directory(directory):
#         images = []
#         for root, dirs, files in os.walk(directory):
#             for file in files:
#                 image_path = os.path.join(root, file)
#                 img = Image.open(image_path).convert('L').resize((img_cols, img_rows))
#                 images.append(np.array(img).flatten())
#         return images

#     # Load images for abnormal (fall) class
#     abnormal_images = load_images_from_directory("C:/Users/admin/Desktop/tryproject/abnormall")
#     abnormal_labels = np.ones(len(abnormal_images))

#     # Load images for normal (not fall) class
#     normal_images = load_images_from_directory("C:/Users/admin/Desktop/tryproject/normal")
#     normal_labels = np.zeros(len(normal_images))

#     # Combine images and labels
#     images = np.vstack((abnormal_images, normal_images))
#     labels = np.hstack((abnormal_labels, normal_labels))

#     # Shuffle the data
#     images, labels = shuffle(images, labels, random_state=2)

#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=1)

#     # Reshape and normalize the data
#     X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 1).astype('float32') / 255
#     X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 1).astype('float32') / 255

#     # Convert labels to one-hot encoding
#     y_train = to_categorical(y_train, num_classes=2)
#     y_test = to_categorical(y_test, num_classes=2)

#     # Define the CNN model
#     model = Sequential([
#         Conv2D(64, kernel_size=3, activation='relu', input_shape=(img_cols, img_rows, 1)),
#         Conv2D(32, kernel_size=3, activation='relu'),
#         Flatten(),
#         Dense(2, activation='softmax')
#     ])

#     # Compile the model
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     # Train the model
#     model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

#     # Evaluate the model
#     _, accuracy = model.evaluate(X_test, y_test)
#     print("Test Accuracy:", accuracy)

#        # Predict class probabilities
#     y_pred_probs = model.predict(X_test)

#     # Convert predicted probabilities to class labels
#     y_pred_classes = np.argmax(y_pred_probs, axis=1)

#     # Convert one-hot encoded y_test back to class labels (0 or 1)
#     y_true = np.argmax(y_test, axis=1)

#     # Import classification metrics
#     from sklearn.metrics import classification_report, confusion_matrix

#     # Print classification report and confusion matrix
#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred_classes, target_names=["Normal", "Abnormal"]))

#     print("Confusion Matrix:")
#     print(confusion_matrix(y_true, y_pred_classes))

#     # Save the model
#     model.save("modelll.h5")

# if __name__ == "__main__":
#     main()

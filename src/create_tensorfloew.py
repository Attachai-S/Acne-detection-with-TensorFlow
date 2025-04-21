import tensorflow as tf
import os
import cv2 as cv
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
# from tensorflow.keras.utils import plot_model

start_time = time.time()  # Record the start time
'''manual install library
pip install tensorflow
winget install graphviz
pip install pydot graphviz : for plot model graph for summary model
'''
print(f"TensorFlow version : {tf.__version__}")
print(f"\n{"="*100}\n")


input_size = 640

def load_dataset_from_folder(folder_path):
    csv_path = os.path.join(folder_path, "_annotations.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์: {csv_path}")

    df = pd.read_csv(csv_path)

    # ✅ Map class string → integer
    class_map = {
        'papules': 0,
        'nodules and cysts': 1,
        'pustules': 2,
        'whitehead and blackhead': 3
    }

    # แปลง class name เป็น integer column ใหม่
    df['class_id'] = df['class'].map(class_map)

    grouped = df.groupby('filename')
    X, Y = [], []

    for filename, group in grouped:
        img_path = os.path.join(folder_path, filename)
        img = cv.imread(img_path)
        if img is None:
            print(f"❌ ไม่พบรูปภาพ: {img_path}")
            continue

        # Normalize และ resize
        img = img.astype(np.float32) / 255.0
        img = cv.resize(img, (input_size, input_size))

        boxes = []
        for _, row in group.iterrows():
            img_w, img_h = row['width'], row['height']
            xmin = row['xmin'] / img_w
            xmax = row['xmax'] / img_w
            ymin = row['ymin'] / img_h
            ymax = row['ymax'] / img_h
            class_id = row['class_id']  # ✅ ใช้ column ที่แปลงแล้ว

            boxes.append([ymin, xmin, ymax, xmax, class_id])

        boxes = np.array(boxes, dtype=np.float32)
        X.append(img)
        Y.append(boxes)

    X_tensor = tf.convert_to_tensor(np.array(X), dtype=tf.float32)
    Y_tensor = tf.ragged.constant(Y, dtype=tf.float32)

    return tf.data.Dataset.from_tensor_slices((X_tensor, Y_tensor))

base_path = r"C:\Users\vangu\PycharmProjects\Acne-detection-wth-TensorFlow\src\Acne04-Detection-5"

train_dataset = load_dataset_from_folder(os.path.join(base_path, "train"))
test_dataset = load_dataset_from_folder(os.path.join(base_path, "test"))
valid_dataset = load_dataset_from_folder(os.path.join(base_path, "valid"))

CLASSES = 4

def format_instance(image, label):
    class_id = tf.one_hot(tf.cast(label[4], tf.int32), CLASSES)
    box = [label[0], label[1], label[2], label[3]]  # normalized box
    return image, (class_id, box)

BATCH_SIZE = 32

# see https://www.tensorflow.org/guide/data_performance

def tune_training_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    dataset = dataset.repeat() # The dataset be repeated indefinitely.
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = tune_training_ds(train_dataset)


def tune_validation_ds(dataset, validation_folder=r"C:\Users\vangu\PycharmProjects\Acne-detection-wth-TensorFlow\src\Acne04-Detection-5\valid"):
    # อ่านไฟล์ _annotation.csv
    csv_path = os.path.join(validation_folder, "_annotations.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # ดึงเฉพาะ filename ที่ไม่ซ้ำ เพื่อให้รู้จำนวนภาพทั้งหมด
    unique_files = df['filename'].nunique()

    # ใช้ batch = 1/4 ของจำนวนภาพ
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(unique_files // 4)
    dataset = dataset.repeat()

    return dataset

validation_ds = tune_validation_ds(valid_dataset)

DROPOUT_FACTOR = 0.5


def build_feature_extractor(inputs):
    # x = tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=(input_size, input_size, 1))(inputs)
    x = tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu')(inputs)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_FACTOR)(x)
    x = tf.keras.layers.AveragePooling2D(2, 2)(x)

    return x


def build_model_adaptor(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    return x


def build_classifier_head(inputs):
    return tf.keras.layers.Dense(CLASSES, activation='softmax', name='classifier_head')(inputs)


def build_regressor_head(inputs):
    return tf.keras.layers.Dense(units=4, name='regressor_head')(inputs)


def build_model(inputs):
    feature_extractor = build_feature_extractor(inputs)

    model_adaptor = build_model_adaptor(feature_extractor)

    classification_head = build_classifier_head(model_adaptor)

    regressor_head = build_regressor_head(model_adaptor)

    model = tf.keras.Model(inputs=inputs, outputs=[classification_head, regressor_head])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss={'classifier_head': 'categorical_crossentropy', 'regressor_head': 'mse'},
                  metrics={'classifier_head': 'accuracy', 'regressor_head': 'mse'})

    return model

# model = build_model(tf.keras.layers.Input(shape=(input_size, input_size, 1,)))
model = build_model(tf.keras.Input(shape=(640, 640, 1)))
model.summary()

# plot_model(model, show_shapes=True, show_layer_names=True)
EPOCHS = 100
train_folder_path = r"C:\Users\vangu\PycharmProjects\Acne-detection-wth-TensorFlow\src\Acne04-Detection-5\train"
image_extensions = ['.jpg', '.jpeg', '.png']
training_file_count = sum(
    1 for file in os.listdir(train_folder_path)
    if os.path.splitext(file)[1].lower() in image_extensions
)
steps_per_epoch = training_file_count // BATCH_SIZE

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        with tf.device('/GPU:0'):
            history = model.fit(
                train_ds,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_ds,
                validation_steps=1,
                epochs=EPOCHS,
                verbose=0
            )
    except RuntimeError as e:
        print(f"❌ เกิดข้อผิดพลาดในการตั้งค่า GPU: {e}")
else:
    print("⚠️ ไม่พบ GPU, กำลังรันบน CPU")
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_ds,
        validation_steps=1,
        epochs=EPOCHS,
        verbose=0
    )
print(f"\n{"="*100}\n")
end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
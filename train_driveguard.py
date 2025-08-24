# #   ## ✅ Final Validation Accuracy: 54.28%
# # # # import pandas as pd
# # # # import os
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # # # from sklearn.model_selection import train_test_split
# # # # import shutil

# # # # # ===== Paths =====
# # # # dataset_dir = "dataset"
# # # # train_dir = os.path.join(dataset_dir, "imgs", "train")
# # # # csv_path = os.path.join(dataset_dir, "driver_imgs_list.csv")

# # # # # ===== Read CSV =====
# # # # df = pd.read_csv(csv_path)

# # # # # Split by driver to avoid leakage
# # # # unique_drivers = df['subject'].unique()
# # # # train_drivers, val_drivers = train_test_split(unique_drivers, test_size=0.2, random_state=42)

# # # # train_df = df[df['subject'].isin(train_drivers)]
# # # # val_df = df[df['subject'].isin(val_drivers)]

# # # # print(f"Train images: {len(train_df)}, Validation images: {len(val_df)}")

# # # # # ===== Temporary split folders =====
# # # # temp_dir = "temp_driveguard"
# # # # train_split_dir = os.path.join(temp_dir, "train")
# # # # val_split_dir = os.path.join(temp_dir, "val")

# # # # # Clean old split folders if they exist
# # # # shutil.rmtree(temp_dir, ignore_errors=True)

# # # # # Create directories
# # # # for split_dir in [train_split_dir, val_split_dir]:
# # # #     for cls in sorted(df['classname'].unique()):
# # # #         os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

# # # # # Copy images into respective split folders
# # # # def copy_images(split_df, split_dir):
# # # #     for _, row in split_df.iterrows():
# # # #         src = os.path.join(train_dir, row['classname'], row['img'])
# # # #         dst = os.path.join(split_dir, row['classname'], row['img'])
# # # #         shutil.copy(src, dst)

# # # # print("Copying training images...")
# # # # copy_images(train_df, train_split_dir)
# # # # print("Copying validation images...")
# # # # copy_images(val_df, val_split_dir)

# # # # # ===== Image Generators =====
# # # # IMG_SIZE = 128
# # # # BATCH_SIZE = 32

# # # # train_datagen = ImageDataGenerator(rescale=1./255)
# # # # val_datagen = ImageDataGenerator(rescale=1./255)

# # # # train_generator = train_datagen.flow_from_directory(
# # # #     train_split_dir,
# # # #     target_size=(IMG_SIZE, IMG_SIZE),
# # # #     batch_size=BATCH_SIZE,
# # # #     class_mode='categorical'
# # # # )

# # # # val_generator = val_datagen.flow_from_directory(
# # # #     val_split_dir,
# # # #     target_size=(IMG_SIZE, IMG_SIZE),
# # # #     batch_size=BATCH_SIZE,
# # # #     class_mode='categorical'
# # # # )

# # # # # ===== CNN Model =====
# # # # model = tf.keras.Sequential([
# # # #     tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
# # # #     tf.keras.layers.MaxPooling2D((2,2)),

# # # #     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
# # # #     tf.keras.layers.MaxPooling2D((2,2)),

# # # #     tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
# # # #     tf.keras.layers.MaxPooling2D((2,2)),

# # # #     tf.keras.layers.Flatten(),
# # # #     tf.keras.layers.Dense(128, activation='relu'),
# # # #     tf.keras.layers.Dropout(0.5),
# # # #     tf.keras.layers.Dense(10, activation='softmax')
# # # # ])

# # # # model.compile(optimizer='adam',
# # # #               loss='categorical_crossentropy',
# # # #               metrics=['accuracy'])

# # # # # ===== Train =====
# # # # history = model.fit(
# # # #     train_generator,
# # # #     validation_data=val_generator,
# # # #     epochs=10
# # # # )

# # # # # ===== Final Accuracy =====
# # # # loss, acc = model.evaluate(val_generator)
# # # # print(f"\n✅ Final Validation Accuracy: {acc*100:.2f}%")

# # # import pandas as pd
# # # import os
# # # import numpy as np
# # # import tensorflow as tf
# # # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # # from sklearn.model_selection import train_test_split
# # # import shutil
# # # from tensorflow.keras.callbacks import EarlyStopping

# # # # ===== Paths =====
# # # dataset_dir = "dataset"
# # # train_dir = os.path.join(dataset_dir, "imgs", "train")
# # # csv_path = os.path.join(dataset_dir, "driver_imgs_list.csv")

# # # # ===== Read CSV =====
# # # df = pd.read_csv(csv_path)

# # # # Split by driver to avoid leakage
# # # unique_drivers = df['subject'].unique()
# # # train_drivers, val_drivers = train_test_split(unique_drivers, test_size=0.2, random_state=42)

# # # train_df = df[df['subject'].isin(train_drivers)]
# # # val_df = df[df['subject'].isin(val_drivers)]

# # # print(f"Train images: {len(train_df)}, Validation images: {len(val_df)}")

# # # # ===== Temporary split folders =====
# # # temp_dir = "temp_driveguard"
# # # train_split_dir = os.path.join(temp_dir, "train")
# # # val_split_dir = os.path.join(temp_dir, "val")

# # # shutil.rmtree(temp_dir, ignore_errors=True)  # Remove old
# # # for split_dir in [train_split_dir, val_split_dir]:
# # #     for cls in sorted(df['classname'].unique()):
# # #         os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

# # # def copy_images(split_df, split_dir):
# # #     for _, row in split_df.iterrows():
# # #         src = os.path.join(train_dir, row['classname'], row['img'])
# # #         dst = os.path.join(split_dir, row['classname'], row['img'])
# # #         shutil.copy(src, dst)

# # # print("Copying training images...")
# # # copy_images(train_df, train_split_dir)
# # # print("Copying validation images...")
# # # copy_images(val_df, val_split_dir)

# # # # ===== Image Generators =====
# # # IMG_SIZE = 128
# # # BATCH_SIZE = 32

# # # train_datagen = ImageDataGenerator(
# # #     rescale=1./255,
# # #     rotation_range=15,
# # #     width_shift_range=0.1,
# # #     height_shift_range=0.1,
# # #     shear_range=0.1,
# # #     zoom_range=0.2,
# # #     horizontal_flip=True
# # # )
# # # val_datagen = ImageDataGenerator(rescale=1./255)

# # # train_generator = train_datagen.flow_from_directory(
# # #     train_split_dir,
# # #     target_size=(IMG_SIZE, IMG_SIZE),
# # #     batch_size=BATCH_SIZE,
# # #     class_mode='categorical'
# # # )
# # # val_generator = val_datagen.flow_from_directory(
# # #     val_split_dir,
# # #     target_size=(IMG_SIZE, IMG_SIZE),
# # #     batch_size=BATCH_SIZE,
# # #     class_mode='categorical'
# # # )

# # # # ===== CNN Model =====
# # # model = tf.keras.Sequential([
# # #     tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
# # #     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
# # #     tf.keras.layers.MaxPooling2D((2,2)),
# # #     tf.keras.layers.Dropout(0.25),

# # #     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
# # #     tf.keras.layers.MaxPooling2D((2,2)),
# # #     tf.keras.layers.Dropout(0.25),

# # #     tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
# # #     tf.keras.layers.MaxPooling2D((2,2)),
# # #     tf.keras.layers.Dropout(0.25),

# # #     tf.keras.layers.Flatten(),
# # #     tf.keras.layers.Dense(128, activation='relu'),
# # #     tf.keras.layers.Dropout(0.5),
# # #     tf.keras.layers.Dense(10, activation='softmax')
# # # ])

# # # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
# # #               loss='categorical_crossentropy',
# # #               metrics=['accuracy'])

# # # # ===== Callbacks =====
# # # early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# # # # ===== Train =====
# # # history = model.fit(
# # #     train_generator,
# # #     validation_data=val_generator,
# # #     epochs=30,
# # #     callbacks=[early_stop]
# # # )

# # # # ===== Final Accuracy =====
# # # loss, acc = model.evaluate(val_generator)
# # # print(f"\n✅ Final Validation Accuracy: {acc*100:.2f}%")

# # # # Optional: clean temp folder
# # # shutil.rmtree(temp_dir, ignore_errors=True)



# # import pandas as pd
# # import os
# # import shutil
# # import tensorflow as tf
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # from sklearn.model_selection import train_test_split
# # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# # # ===== Paths =====
# # dataset_dir = "dataset"
# # train_dir = os.path.join(dataset_dir, "imgs", "train")
# # csv_path = os.path.join(dataset_dir, "driver_imgs_list.csv")

# # # ===== Read CSV =====
# # df = pd.read_csv(csv_path)

# # # Split by driver to avoid leakage
# # unique_drivers = df['subject'].unique()
# # train_drivers, val_drivers = train_test_split(unique_drivers, test_size=0.2, random_state=42)

# # train_df = df[df['subject'].isin(train_drivers)]
# # val_df = df[df['subject'].isin(val_drivers)]

# # print(f"Train images: {len(train_df)}, Validation images: {len(val_df)}")

# # # ===== Temporary split folders =====
# # temp_dir = "temp_driveguard"
# # train_split_dir = os.path.join(temp_dir, "train")
# # val_split_dir = os.path.join(temp_dir, "val")

# # shutil.rmtree(temp_dir, ignore_errors=True)  # Remove old
# # for split_dir in [train_split_dir, val_split_dir]:
# #     for cls in sorted(df['classname'].unique()):
# #         os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

# # def copy_images(split_df, split_dir):
# #     for _, row in split_df.iterrows():
# #         src = os.path.join(train_dir, row['classname'], row['img'])
# #         dst = os.path.join(split_dir, row['classname'], row['img'])
# #         shutil.copy(src, dst)

# # print("Copying training images...")
# # copy_images(train_df, train_split_dir)
# # print("Copying validation images...")
# # copy_images(val_df, val_split_dir)

# # # ===== Image Generators =====
# # IMG_SIZE = 224  # MobileNetV2 default input size
# # BATCH_SIZE = 32

# # train_datagen = ImageDataGenerator(
# #     rescale=1./255,
# #     rotation_range=20,
# #     width_shift_range=0.2,
# #     height_shift_range=0.2,
# #     shear_range=0.2,
# #     zoom_range=0.2,
# #     horizontal_flip=True,
# #     fill_mode='nearest'
# # )
# # val_datagen = ImageDataGenerator(rescale=1./255)

# # train_generator = train_datagen.flow_from_directory(
# #     train_split_dir,
# #     target_size=(IMG_SIZE, IMG_SIZE),
# #     batch_size=BATCH_SIZE,
# #     class_mode='categorical'
# # )
# # val_generator = val_datagen.flow_from_directory(
# #     val_split_dir,
# #     target_size=(IMG_SIZE, IMG_SIZE),
# #     batch_size=BATCH_SIZE,
# #     class_mode='categorical'
# # )

# # # ===== Transfer Learning Model =====
# # base_model = tf.keras.applications.MobileNetV2(
# #     weights='imagenet', 
# #     include_top=False, 
# #     input_shape=(IMG_SIZE, IMG_SIZE, 3)
# # )
# # base_model.trainable = False  # freeze pretrained layers

# # model = tf.keras.Sequential([
# #     base_model,
# #     tf.keras.layers.GlobalAveragePooling2D(),
# #     tf.keras.layers.Dropout(0.5),
# #     tf.keras.layers.Dense(128, activation='relu'),
# #     tf.keras.layers.Dropout(0.5),
# #     tf.keras.layers.Dense(10, activation='softmax')
# # ])

# # model.compile(
# #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
# #     loss='categorical_crossentropy',
# #     metrics=['accuracy']
# # )

# # # ===== Callbacks =====
# # early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
# # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# # # ===== Train =====
# # history = model.fit(
# #     train_generator,
# #     validation_data=val_generator,
# #     epochs=30,
# #     callbacks=[early_stop, reduce_lr]
# # )

# # # ===== Final Accuracy =====
# # loss, acc = model.evaluate(val_generator)
# # print(f"\n✅ Final Validation Accuracy: {acc*100:.2f}%")   ##Final Validation Accuracy:46.03%

# # # Optional: clean temp folder
# # shutil.rmtree(temp_dir, ignore_errors=True)





# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# import os

# # =========================
# # Paths
# # =========================
# dataset_dir = "dataset/imgs/train"  # replace with your dataset path

# # =========================
# # Data Preparation
# # =========================
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
#     zoom_range=0.1,
#     validation_split=0.2  # 20% for validation
# )

# train_generator = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     subset='training'
# )

# val_generator = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation'
# )

# # =========================
# # Build Model (EfficientNet)
# # =========================
# base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
# base_model.trainable = False  # freeze base initially

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.3)(x)
# output = Dense(train_generator.num_classes, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=output)

# model.compile(
#     optimizer=Adam(learning_rate=1e-4),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # =========================
# # Callbacks
# # =========================
# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
# early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# # =========================
# # Train Model
# # =========================
# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=30,
#     callbacks=[lr_scheduler, early_stop]
# )

# # =========================
# # Fine-Tuning (Optional)
# # =========================
# base_model.trainable = True
# for layer in base_model.layers[:-50]:  # freeze first layers, fine-tune last layers
#     layer.trainable = False

# model.compile(
#     optimizer=Adam(learning_rate=1e-5),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# history_finetune = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=15,
#     callbacks=[lr_scheduler, early_stop]
# )

# # =========================
# # Save Model
# # =========================
# model.save("DriveGuard_EfficientNet.h5")

# print("✅ Training complete. Model saved as DriveGuard_EfficientNet.h5")


# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# import os

# # =========================
# # Paths
# # =========================
# dataset_dir = "dataset/imgs/train"  # replace with your dataset path

# # =========================
# # Data Preparation
# # =========================
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
#     zoom_range=0.1,
#     validation_split=0.2  # 20% for validation
# )

# train_generator = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',  # for multi-class
#     subset='training'
# )

# val_generator = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',  # for multi-class
#     subset='validation'
# )

# # =========================
# # Build Model (EfficientNet)
# # =========================
# base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
# base_model.trainable = False  # freeze base initially

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.3)(x)
# output = Dense(train_generator.num_classes, activation='softmax')(x)  # 10 neurons for 10 classes

# model = Model(inputs=base_model.input, outputs=output)

# model.compile(
#     optimizer=Adam(learning_rate=1e-4),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # =========================
# # Callbacks
# # =========================
# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
# early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# # =========================
# # Train Model
# # =========================
# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=30,
#     callbacks=[lr_scheduler, early_stop]
# )

# # =========================
# # Fine-Tuning (Optional)
# # =========================
# base_model.trainable = True
# for layer in base_model.layers[:-50]:  # freeze first layers, fine-tune last layers
#     layer.trainable = False

# model.compile(
#     optimizer=Adam(learning_rate=1e-5),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# history_finetune = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=15,
#     callbacks=[lr_scheduler, early_stop]
# )

# # =========================
# # Save Model
# # =========================
# model.save("DriveGuard_EfficientNet10.h5")

# print("✅ Training complete. Model saved as DriveGuard_EfficientNet10.h5")

# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# from tensorflow.keras import mixed_precision
# import os

# # =========================
# # Mixed Precision for Faster Training
# # =========================
# mixed_precision.set_global_policy('mixed_float16')

# # =========================
# # Paths
# # =========================
# dataset_dir = "dataset/imgs/train"  # Replace with your path

# # =========================
# # Data Preparation
# # =========================
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
#     zoom_range=0.1,
#     brightness_range=[0.8,1.2],
#     validation_split=0.2
# )

# train_generator = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=(224,224),
#     batch_size=32,
#     class_mode='categorical',
#     subset='training'
# )

# val_generator = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=(224,224),
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation'
# )

# # =========================
# # Build Model
# # =========================
# base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
# base_model.trainable = False  # Freeze base

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.3)(x)
# output = Dense(train_generator.num_classes, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=output)

# model.compile(
#     optimizer=Adam(learning_rate=1e-4),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # =========================
# # Callbacks
# # =========================
# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
# early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# # =========================
# # Train Frozen Base
# # =========================
# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=15,
#     callbacks=[lr_scheduler, early_stop]
# )

# # =========================
# # Fine-Tuning Last Layers
# # =========================
# base_model.trainable = True
# for layer in base_model.layers[:-30]:
#     layer.trainable = False  # Freeze first layers, fine-tune last 30

# model.compile(
#     optimizer=Adam(learning_rate=1e-5),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# history_finetune = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=15,
#     callbacks=[lr_scheduler, early_stop]
# )

# # =========================
# # Save Model
# # =========================
# model.save("DriveGuard_EfficientNet10.h5")
# print("✅ Model saved successfully!")



# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# import os
# import sys

# # =========================
# # Check GPU availability
# # =========================
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print(f"✅ GPU detected: {gpus}")
# else:
#     print("⚠️ No GPU detected. Training may be slow.")

# # =========================
# # Mixed Precision (Optional)
# # =========================
# try:
#     from tensorflow.keras import mixed_precision
#     mixed_precision.set_global_policy('mixed_float16')
#     print("✅ Mixed precision enabled")
# except Exception as e:
#     print("⚠️ Mixed precision not enabled:", e)

# # =========================
# # Dataset Path
# # =========================
# dataset_dir = "dataset/imgs/train"
# if not os.path.exists(dataset_dir):
#     print(f"❌ Dataset path not found: {dataset_dir}")
#     sys.exit()

# print("✅ Dataset path exists. Subfolders:", os.listdir(dataset_dir))

# # =========================
# # Data Generators
# # =========================
# batch_size = 16  # smaller for safer memory usage

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
#     zoom_range=0.1,
#     brightness_range=[0.8,1.2],
#     validation_split=0.2
# )

# train_generator = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=(224,224),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='training'
# )

# val_generator = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=(224,224),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='validation'
# )

# # =========================
# # Build Model
# # =========================
# base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
# base_model.trainable = False  # Freeze base

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.3)(x)
# output = Dense(train_generator.num_classes, activation='softmax', dtype='float32')(x)  # force float32 output

# model = Model(inputs=base_model.input, outputs=output)

# model.compile(
#     optimizer=Adam(learning_rate=1e-4),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # =========================
# # Callbacks
# # =========================
# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
# early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# # =========================
# # Train Frozen Base
# # =========================
# print("🚀 Starting training with frozen base...")
# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=15,
#     callbacks=[lr_scheduler, early_stop]
# )

# # =========================
# # Fine-Tuning Last Layers
# # =========================
# print("🚀 Fine-tuning last 30 layers...")
# base_model.trainable = True
# for layer in base_model.layers[:-30]:
#     layer.trainable = False

# model.compile(
#     optimizer=Adam(learning_rate=1e-5),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# history_finetune = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=15,
#     callbacks=[lr_scheduler, early_stop]
# )

# # =========================
# # Save Model
# # =========================
# model.save("DriveGuard_EfficientNet10.h5")
# print("✅ Model saved successfully!")


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight
import numpy as np
import os
import sys

# =========================
# Check GPU availability
# =========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detected: {gpus}")
else:
    print("⚠️ No GPU detected. Training on CPU will be slow.")

# =========================
# Dataset Path
# =========================
dataset_dir = "dataset/imgs/train"
if not os.path.exists(dataset_dir):
    print(f"❌ Dataset path not found: {dataset_dir}")
    sys.exit()

print("✅ Dataset path exists. Subfolders:", os.listdir(dataset_dir))

# =========================
# Data Generators
# =========================
batch_size = 16  # safer for CPU

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.8,1.2],
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# =========================
# Compute Class Weights
# =========================
labels = train_generator.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
print("✅ Computed class weights:", class_weights)

# =========================
# Build Model
# =========================
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Unfreeze last 100 layers for better adaptation
base_model.trainable = True
for layer in base_model.layers[:-100]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(train_generator.num_classes, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=5e-5),  # slightly higher LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# Callbacks
# =========================
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# =========================
# Train Model
# =========================
print("🚀 Starting training with partial fine-tuning...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,   # longer training
    callbacks=[lr_scheduler, early_stop],
    class_weight=class_weights
)

# =========================
# Save Model
# =========================
model.save("DriveGuard_EfficientNet10.h5")
print("✅ Model saved successfully!")


from body_parts import Body_Parts
import itertools
import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras


def load_pose_landmarks(df):
    """Loads a DataFrame created by DataLoader.
  Returns:
    X: Detected landmark coordinates and scores of shape (N, 300 , 25 * 3)
    y: Ground truth labels of shape (N, label_count)
    classes: The list of all class names found in the dataset
  """

    # Load the CSV file
    dataframe = df
    df_to_process = dataframe.copy()
    df_to_process.to_csv("out.csv")

    # Extract the list of class names
    classes = df_to_process.pop('class_name').unique()

    # shuffle the dataframe. this will ensure each batch in tranining loop has representation from all classes.
    # df_to_process = df_to_process.sample(frac=1).reset_index(drop=True)

    # Extract the labels
    y = df_to_process.pop('class_no')

    # Convert the input features and labels into the correct format for training.
    X = df_to_process.pop('frames').to_numpy()
    #X = np.asarray(X).astype('float32')
    #X = df_to_process.astype('float64')
    y = keras.utils.to_categorical(y)

    return X, y, classes, dataframe


def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""

    left = tf.gather(landmarks, left_bodypart, axis=1)
    right = tf.gather(landmarks, right_bodypart, axis=1)
    center = left * 0.5 + right * 0.5
    return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.

  It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
  """
    # Hips center
    hips_center = get_center_point(landmarks, Body_Parts["LHip"],
                                   Body_Parts["RHip"])

    # Shoulders center
    shoulders_center = get_center_point(landmarks, Body_Parts["LShoulder"],
                                        Body_Parts["RShoulder"])

    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center - hips_center)

    # Pose center
    pose_center_new = get_center_point(landmarks, Body_Parts["LHip"],
                                       Body_Parts["RHip"])
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(pose_center_new,
                                      [tf.size(landmarks) // (25 * 2), 25, 2])

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                  name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

    return pose_size


def normalize_pose_landmarks(landmarks):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
  scaling it to a constant pose size.
  """
    # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, Body_Parts["LHip"],
                                   Body_Parts["RHip"])
    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center,
                                  [tf.size(landmarks) // (25 * 2), 25, 2])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size

    return landmarks


def landmarks_to_embedding(frames_of_keypoints):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(17, 3)
    #reshaped_inputs = keras.layers.Reshape((300, 25, 3))(frames_of_keypoints)

    # Normalize landmarks 2D
    #landmarks = normalize_pose_landmarks(reshaped_inputs[, :, :, :2])

    # Flatten the normalized landmark coordinates into a vector
    embedding = keras.layers.Flatten()(frames_of_keypoints)

    return embedding


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """Plots the confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

import os
import cv2
import sys
import glob
import time
import dlib
import numpy as np
from tqdm import tqdm

COSINE_THRESHOLD = 0.6
NORML2_THRESHOLD = 1.128

def match(recognizer, feature1, dictionary):
    for element in dictionary:
        user_id, feature2 = element
        score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        print('score match = ', score)

        if score > COSINE_THRESHOLD:
            return True, (user_id, score)
    return False, ("", 0.0)

def crop_face(image, face, file_name):
	# print(file_name)
	x = int(face[0])
	y = int(face[1])
	w = int(face[2])
	h = int(face[3])
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	roi_color = image[y:y + h, x:x + w] 
	if file_name is not None:
		cv2.imwrite('data/temp_dlib/' + file_name + '_face.png', roi_color)
	return roi_color
 
def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() -  rect.top()

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return [x, y, w, h]
    
def recognize_face(image, face_detector, face_recognizer, file_name=None):
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    height, width, _ = image.shape
    
    # face_detector.setInputSize((width, height))

    st = time.time()
    faces = [_trim_css_to_bounds(_rect_to_css(face.rect), image.shape) for face in face_detector(image, 1)]
    print(f'time to run face detection = {time.time() - st}')
    # boxes = [convert_and_trim_bb(image, r.rect) for r in results]
    faces = faces if faces is not None else []
    features = []
    for idx, face in enumerate(faces):
        stf = time.time()
        if file_name is not None:
            # DEBUG: record the cropped face for reviewing
            cropped_face = crop_face(image, faces[0], file_name.split('/')[-1].split('.')[0] + '_0')
        else:
            cropped_face = crop_face(image, faces[0], None)
 
        # aligned_face = face_recognizer.alignCrop(image, face)
        if file_name is not None:
                # DEBUG: record the cropped face for reviewing
                # aligned_face = crop_face(image, faces[0], file_name.split('/')[-1].split('.')[0] + '_0')
                # 画像を表示、保存する
                # for i, aligned_face in enumerate(aligned_face):
            cv2.imwrite('data/temp_aligned_dlib/' + file_name.split('/')[-1].split('.')[0] + '_0' + '_face.png', cropped_face)
        else:
            cv2.imwrite('data/temp_aligned_dlib/' + '_0' + '_face.png', cropped_face)

        strd = time.time()
        feat = face_recognizer.feature(cropped_face)
        print(f'time to run face features recognizer = {time.time() - strd}')

        features.append(feat)
    return features, faces

def main():
    # contain npy for embedings and registration photos
    directory = 'data'
    
    # Init models face detection & recognition
    weights = os.path.join(directory, "models", "mmod_human_face_detector.dat")
    face_detector = dlib.cnn_face_detection_model_v1(weights)
    
    weights = os.path.join(directory, "models", "face_recognizer_fast.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")
    
    # capture = cv2.VideoCapture(os.path.join(directory, "image.jpg")) # 画像ファイル
    capture = cv2.VideoCapture(0) # カメラ
    if not capture.isOpened():
        exit()
    
    # Get registered photos and return as npy files
    # File name = id name, embeddings of a photo is the representative for the id
    # If many files have the same name, an average embedding is used
    dictionary = []
    # the tuple of file types, please ADD MORE if you want

    types = ('*.jpg', '*.png', '*.jpeg') 
    files = []
    for type in types:
        files.extend(glob.glob(os.path.join(directory, 'test', type)))

    for file in tqdm(files):
        image = cv2.imread(file)

        feats, faces = recognize_face(image, face_detector, face_recognizer, file)
        user_id = os.path.splitext(os.path.basename(file))[0]
        dictionary.append((user_id, feats[0]))
    
    print(f'there are {len(dictionary)} ids')
    while True:
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        fetures, faces = recognize_face(image, face_detector, face_recognizer)
        for idx, (face, feature) in enumerate(zip(faces, fetures)):
            stm = time.time()
            result, user = match(face_recognizer, feature, dictionary)
            print(f'time to run face matching recognizer = {time.time() - stm}')

            box = list(map(int, face[:4]))
            color = (0, 255, 0) if result else (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            id, score = user if result else (f"unknown_{idx}", 0.0)
            text = "{0} ({1:.2f})".format(id, score)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            cv2.putText(image, text, position, font, scale, color, thickness, cv2.LINE_AA)

        cv2.imshow("face recognition", image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
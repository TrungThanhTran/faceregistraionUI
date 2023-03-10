from __future__ import annotations
import json
import os
import gradio as gr
import numpy as np
import cv2
import io
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('Agg')


# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933]
]

TITLE = 'Face Registration ClientScan - Demo'
DESCRIPTION = '<center>This is a web front-end of clientScan</center>'
ARTICLE = '<center><img src="https://clientscan.co.uk" alt=""/></center>'

# Function to run the app
# contain npy for embedings and registration photos
DATA_DIRECTORY = 'data'
REGISTRATION_DIRECTORY = 'registraion'

if os.path.exists(os.path.join(DATA_DIRECTORY, REGISTRATION_DIRECTORY)):
    os.mkdir(REGISTRATION_DIRECTORY)

# Init models face detection & recognition
weights = os.path.join(DATA_DIRECTORY, "models",
                       "face_detection_yunet_2022mar.onnx")
face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
weights = os.path.join(DATA_DIRECTORY, "models", "face_recognizer_fast.onnx")
face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

def set_image(image):
    return gr.Image.update(value=image[0][0])

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    pil_img = Image.open(buf)
    basewidth = 1000
    wpercent = (basewidth/float(pil_img.size[0]))
    hsize = int((float(pil_img.size[1])*float(wpercent)))
    img = pil_img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
    return img

def crop_face_aligned_face(aligned_face, name_file):
    # print(file_name)
    cv2.imwrite(os.path.join(DATA_DIRECTORY, 'registration',
                name_file + '.png'), aligned_face)

def visualize_prediction(img, faces):
    plt.figure(figsize=(50, 50))
    plt.imshow(img)
    ax = plt.gca()
    colors = COLORS * 100
    for face, color in zip(faces, colors):
        (x, y, h, w) = face[:4]  # list(map(int, face[:4]))
        score = face[-1]
        ax.add_patch(plt.Rectangle(
            (x, y), h, w, fill=False, color=color, linewidth=10))
        ax.text(x, y, f"Confidence: {int(score * 100)}% ", fontsize=55, bbox=dict(
            facecolor="yellow", alpha=0.5))
    plt.axis("off")
    return fig2img(plt.gcf())

def visualize_cv2(image, faces):
    for idx, face in enumerate(faces):
        box = list(map(int, face[:4]))
        color = (0, 255, 0) 
        thickness = 2
        cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

        text = f"Confidence : {face[-1]}"
        position = (box[0], box[1] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        cv2.putText(image, text, position, font, scale,
                    color, thickness, cv2.LINE_AA)
    return image

def get_biggest_face(faces):
    max_index = 0
    max_size = 0.0
    for idx, face in enumerate(faces):
        if face[2] * face[3] > max_size:
            max_size = face[2] * face[3]
            max_index = idx
    return max_index

def detect(image,
           face_score_threshold,
           name_box,
           id_box):

    # Gradio.Image read in numpy Image, which in channels = BRG
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reset face detection score threshold
    face_detector.setScoreThreshold(float(face_score_threshold))

    # Check image channels
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
    image = cv2.resize(image, (0, 0),
                fx=(500 / image.shape[0]), fy=(500 / image.shape[0]))

    # detected_faces  = {}
    # # THE FIRSt round for full image 
    # ratios = [image.shape[0], 500, 1000]
    # for idx, rate in enumerate(ratios):
    #     image = cv2.resize(image, (0, 0),
    #             fx=(rate / image.shape[0]), fy=(rate / image.shape[0]))

    #     height, width, _ = image.shape
    #     face_detector.setInputSize((width, height))

    #     # FACE DETECTION
    #     _, faces = face_detector.detect(image)
    #     if faces is None:
    #         continue
        
    #     # Get biggest one
    #     max_idx = get_biggest_face(faces)
    #     detected_faces[idx] = faces[max_idx]
    # max_conf = 0.0
    # used_face = None
    # rate_used = 0
    # for rate_idx, face in zip(detected_faces.keys(), detected_faces.values()):
    #     if face[-1] > max_conf:
    #         max_conf = face[-1]
    #         used_face = face
    #         rate_used = rate_idx
    
    # print('size of image = ', used_face[2] * used_face[3])

    # used_image = cv2.resize(image, (0, 0),
    #             fx=(ratios[rate_used] / image.shape[0]), fy=(ratios[rate_used] / image.shape[0]))
    
    height, width, _ = image.shape
    face_detector.setInputSize((width, height))

    # FACE DETECTION
    _, faces = face_detector.detect(image)
    max_idx = get_biggest_face(faces)

    print('size of image = ', faces[max_idx][2] * faces[max_idx][3])

    aligned_face = face_recognizer.alignCrop(image, faces[max_idx])

    # Write cropped image
    crop_face_aligned_face(aligned_face, f'{name_box}_{id_box}')

    # FACE RECOGNITION
    feat = face_recognizer.feature(aligned_face)

    # Data to be written
    file_feat = os.path.join(
        DATA_DIRECTORY, 'registration', f'{name_box}_{id_box}.npy')
    np.save(file_feat, feat)

    # ??Write metadata
    dictionary = {
        "name": name_box,
        "id_number": id_box,
        "feat_file": file_feat
    }

    # Serializing json
    json_object = json.dumps(dictionary, indent=4)

    # Writing to sample.json
    with open(os.path.join(DATA_DIRECTORY, 'registration', f'{name_box}_{id_box}.json'), "w") as outfile:
        outfile.write(json_object)

    # assert len(faces) == 1, "THE PHOTO SHOULD CONTAIN ONLY 1 FACE"
    
    visual_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    showed_image = visualize_prediction(visual_image, faces)
    # showed_image = visualize_cv2(image, faces)
    return showed_image


def interface() -> None:
    """
    Create and launch the graphical user interface face detection app.
    """
    # Create the blocks for the interface
    with gr.Blocks() as app:
        # Add a title and opening HTML element
        gr.HTML(
            """
            <div style="text-align: center; max-width: 650px; margin: 0 auto; padding-top: 7px;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.85rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Face Registration ClientScan - Demo
                </h1>
              </div>
            </div>
        """
        )
        with gr.Group():
            with gr.Column():
                name_box = gr.Textbox(
                    label="Enter your Name",
                    lines=1,
                )
            with gr.Column():
                id_box = gr.Textbox(
                    label="Enter your id number",
                    lines=1,
                )
            with gr.Tabs():
                with gr.TabItem("Image input ???????"):
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                image_in = gr.Image(
                                    label="Image input", interactive=True)
                            with gr.Row():
                                score_slider = gr.Slider(0, 1, value=0.9, step=0.01,
                                                         label='Face Score Threshold')
                            with gr.Row():
                                detect_button = gr.Button(
                                    value="Detect face ????")
                            with gr.Row():
                                paths = [["examples/" + example]
                                         for example in os.listdir("examples")
                                         if '.jpg' in example]
                                example_images = gr.Dataset(
                                    components=([image_in]),
                                    label="Example images",
                                    samples=[path for path in paths])
                        with gr.Column():
                            with gr.Row():
                                face_detected_image_out = gr.Image(
                                    label="Face detected")

                            example_images.click(fn=set_image, inputs=[
                                                 example_images], outputs=[image_in])

                            detect_button.click(fn=detect,
                                                inputs=[
                                                    image_in, score_slider, name_box, id_box],
                                                outputs=face_detected_image_out,
                                                show_progress=True,
                                                queue=True)
                with gr.TabItem("Camera input ????"):
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                webcam_image_in = gr.Webcam(
                                    label="Webcam input")
                            with gr.Row():
                                score_slider = gr.Slider(0, 1, value=0.9, step=0.01,
                                                         label='Face Score Threshold')
                            with gr.Row():
                                detect_button = gr.Button(
                                    value="Detect face ????")
                        with gr.Column():
                            with gr.Row():
                                face_detected_webcam_out = gr.Image(
                                    label="Face detected")
                            detect_button.click(fn=detect,
                                                inputs=[
                                                    webcam_image_in, score_slider, name_box, id_box],
                                                outputs=face_detected_webcam_out)
        # Set queue operation without allowing to call API request continously
        app.launch(
            enable_queue=True,
            server_port=1234,
            share=True,
            show_error=True)

    gr.close_all()


if __name__ == '__main__':
    interface()  # Run the interface

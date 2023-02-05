This is a prototype for a real-time face recognition implemented by Trung Tran(tranthanhtrung1990@gmail.com) in Python. This is the code only for the project ClientScan authorized by Robin Paine.


# Update
31/01/2023: Finish the prototype face recognition using Yunet + Sface
05/02/2023: Add web interface for face registration check 
[New]:
    - Installing gradion in # Intsallation
    - Check # Run front-end web app

# Environment
Requirements: python 3.6.8 for dlib 

# Intsallation
pip install dlib
pip install tqdm
pip install opencv-python==4.7.0
pip install gradio==3.1.4

# Run face detection = Yunet & face recognition = SFace
python face_recognizer_yunet.py

# Run face detection = Dlib & face recognition = SFace
python face_recognizer_dlib.py

# Run front-end web app: 
python front_end_web.py
![Web Interface](data/result/web_demo.png)
Open localhost:1234 or 127.0.0.1:1234
Or access the public address display in the terminal
for example: https://23024e95c2f96878.gradio.app


# Data preparation
Add profile/register photo into folder `data/images` as name_of_person.file_types
support file_types = 'jpg', 'png', 'jpeg'. Please add more if you want by searching line of code: `types = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')`

# Reviewing
Add recording cropped face in temp folder for reviewing

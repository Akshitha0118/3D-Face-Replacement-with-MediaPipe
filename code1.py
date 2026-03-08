import cv2
import mediapipe as mp
import numpy as np

# -------- MEDIAPIPE --------
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5
)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, color=(0,255,0))


# -------- FUNCTIONS --------
def get_face_bbox(landmarks, image_shape):

    x = [lm[0] for lm in landmarks]
    y = [lm[1] for lm in landmarks]

    xmin = int(min(x) * image_shape[1])
    ymin = int(min(y) * image_shape[0])
    xmax = int(max(x) * image_shape[1])
    ymax = int(max(y) * image_shape[0])

    return xmin, ymin, xmax, ymax


def transform_3d_face(image, landmarks, replacement_face):

    transformed_image = image.copy()

    xmin, ymin, xmax, ymax = get_face_bbox(landmarks, image.shape[:2])

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(image.shape[1], xmax)
    ymax = min(image.shape[0], ymax)

    if xmax-xmin <= 0 or ymax-ymin <= 0:
        return transformed_image

    resized_face = cv2.resize(replacement_face,(xmax-xmin,ymax-ymin))

    mask = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY) / 255.0

    roi = transformed_image[ymin:ymax, xmin:xmax]

    roi = roi*(1-mask[:,:,np.newaxis]) + resized_face*(mask[:,:,np.newaxis])

    transformed_image[ymin:ymax, xmin:xmax] = roi

    return transformed_image


# -------- INPUT VIDEO --------
video_path = r"C:\Users\Admin\Desktop\MEDIAPIPE\3d_face_onlyface\From Main Klickpin CF- Pinterest Video - 5wHiHcO5H.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()


# -------- REPLACEMENT FACE IMAGE --------
replacement_face = cv2.imread("face.jpg")

if replacement_face is None:
    print("Error: face.jpg not found")
    exit()


# -------- OUTPUT VIDEO --------
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (frame_width, frame_height))


# -------- PROCESS VIDEO --------
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    transformed_image = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            landmarks = [(lm.x,lm.y,lm.z) for lm in face_landmarks.landmark]

            transformed_image = transform_3d_face(frame, landmarks, replacement_face)

            mp_drawing.draw_landmarks(
                transformed_image,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    out.write(transformed_image)

    cv2.imshow("Face Transform", transformed_image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved as output_video.mp4")
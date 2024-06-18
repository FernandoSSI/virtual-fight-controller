import cv2
import time
import mediapipe as mp
import numpy as np
import vgamepad as vg

class VirtualFightController:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.gamepad = vg.VX360Gamepad()

        self.punch_count_r = 0
        self.punch_count_l = 0
        self.angle_threshold = 180
        self.cooldown_time = 0.5
        self.last_punch_r = time.time()
        self.last_punch_l = time.time()

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine)
        return np.degrees(angle)

    def points_visible(self, landmarks, indexes):
        for index in indexes:
            if landmarks.landmark[index].visibility < 0.5:
                return False
        return True

    def detect_punch(self, pose_landmarks):
        if self.points_visible(pose_landmarks, [self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                                                self.mp_pose.PoseLandmark.RIGHT_WRIST]):

            shoulder_r = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            elbow_r = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            wrist_r = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y]
            angle_right = self.calculate_angle(shoulder_r, elbow_r, wrist_r)

            if angle_right > 160 and angle_right <= self.angle_threshold and (
                    time.time() - self.last_punch_r) > self.cooldown_time:
                self.punch_count_r += 1
                self.last_punch_r = time.time()
                return "right"

        if self.points_visible(pose_landmarks, [self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                                                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                                                self.mp_pose.PoseLandmark.LEFT_WRIST]):

            shoulder_l = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                          pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            elbow_l = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
            wrist_l = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
            angle_left = self.calculate_angle(shoulder_l, elbow_l, wrist_l)

            if angle_left > 160 and angle_left <= self.angle_threshold and (
                    time.time() - self.last_punch_l) > self.cooldown_time:
                self.punch_count_l += 1
                self.last_punch_l = time.time()
                return "left"

        return None

    def toPunch(self):
        self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        self.gamepad.update()
        time.sleep(0.1)
        self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        self.gamepad.update()

    def run(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                detected_punch = self.detect_punch(results.pose_landmarks)

                if detected_punch:
                    self.toPunch()

            cv2.imshow("Virtual Fight Controller", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = VirtualFightController()
    controller.run()

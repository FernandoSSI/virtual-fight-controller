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

        self.cooldown_time = 0.5
        self.last_punch_r = time.time()
        self.last_punch_l = time.time()
        self.last_walk_r = time.time()
        self.last_walk_l = time.time()

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

            if 160 < angle_right <= 180 and (
                    time.time() - self.last_punch_r) > self.cooldown_time:
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

            if 150 < angle_left <= 180 and (
                    time.time() - self.last_punch_l) > self.cooldown_time:
                self.last_punch_l = time.time()
                return "left"

        return None

    def detect_walk(self, pose_landmarks):
        if self.points_visible(pose_landmarks, [self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                                self.mp_pose.PoseLandmark.RIGHT_HIP,
                                                self.mp_pose.PoseLandmark.RIGHT_KNEE]):
            shoulder_r = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            hip_r = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                     pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y]
            knee_r = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].x,
                      pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].y]
            angle_right = self.calculate_angle(shoulder_r, hip_r, knee_r)

            if 90 < angle_right <= 130 and (
                    time.time() - self.last_walk_r) > self.cooldown_time:
                self.last_walk_r = time.time()
                return "right"

        if self.points_visible(pose_landmarks, [self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                                                self.mp_pose.PoseLandmark.LEFT_HIP,
                                                self.mp_pose.PoseLandmark.LEFT_KNEE]):
            shoulder_l = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                          pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            hip_l = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                     pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            knee_l = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                      pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            angle_left = self.calculate_angle(shoulder_l, hip_l, knee_l)

            if 90 < angle_left <= 130 and (
                    time.time() - self.last_walk_l) > self.cooldown_time:
                self.last_walk_r = time.time()
                return "left"
        return None

    def to_punch(self, arm):
        if arm == "left":
            self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
            self.gamepad.update()
            time.sleep(0.1)
            self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
            self.gamepad.update()
        elif arm == "right":
            self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
            self.gamepad.update()
            time.sleep(0.1)
            self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
            self.gamepad.update()

    def to_walk(self, direction):
        if direction == "right":
            self.gamepad.left_joystick(-32767, 0)
            self.gamepad.update()
            time.sleep(0.3)
            self.gamepad.left_joystick(0, 0)
            self.gamepad.update()
        if direction == "left":
            self.gamepad.left_joystick(32767, 0)
            self.gamepad.update()
            time.sleep(0.3)
            self.gamepad.left_joystick(0, 0)
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
                detected_walk = self.detect_walk(results.pose_landmarks)

                if detected_punch:
                    self.to_punch(detected_punch)

                if detected_walk:
                    self.to_walk(detected_walk)

            cv2.imshow("Virtual Fight Controller", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    controller = VirtualFightController()
    controller.run()

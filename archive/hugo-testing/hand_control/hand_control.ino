#include <Servo.h>

// Servo indexes

Servo servoindex2; // Ring
Servo servoindex3; // Wrist
Servo servoindex4; // Index
Servo servoindex5; // Thumb
Servo servoindex6; // Pinky
Servo servoindex7; // Middle

//Config

#define BAUDRATE 115200
#define MAX_STEP 4
#define WRIST_NEUTRAL 135

// the mechanical limits
#define THUMB_MAX  100
#define INDEX_MAX  110
#define MIDDLE_MAX 90
#define RING_MAX   110
#define PINKY_MAX  100

//State

bool calibrationDone = false;

// Order: THUMB, INDEX, MIDDLE, RING, PINKY, WRIST, UNUSED
int target[7]  = {0, 0, 0, 0, 0, WRIST_NEUTRAL, 90};
int current[7] = {0, 0, 0, 0, 0, WRIST_NEUTRAL, 90};

// Helpers

int clampFinger(int val, int maxVal) {
  return min(max(val, 0), maxVal);
}

int clampWrist(int val) {
  return min(max(val, 0), 180);
}

int stepLimit(int cur, int tgt) {
  if (abs(tgt - cur) <= MAX_STEP) return tgt;
  return cur + (tgt > cur ? MAX_STEP : -MAX_STEP);
}

// Serial (Arduino MEGA)

bool parseFrame(String frame) {
  frame.trim();

  if (frame == "<CALIB_DONE>") {
    calibrationDone = true;
    Serial.println("ACK_CALIB");
    return false;
  }

  if (!frame.startsWith("<") || !frame.endsWith(">")) return false;

  frame.remove(0, 1);
  frame.remove(frame.length() - 1);

  int vals[7];
  int i = 0;

  char buf[80];
  frame.toCharArray(buf, sizeof(buf));
  char *tok = strtok(buf, ",");

  while (tok && i < 7) {
    vals[i++] = atoi(tok);
    tok = strtok(NULL, ",");
  }

  if (i != 7) return false;

  // Apply HARD LIMITS
  target[0] = clampFinger(vals[0], THUMB_MAX);
  target[1] = clampFinger(vals[1], INDEX_MAX);
  target[2] = clampFinger(vals[2], MIDDLE_MAX);
  target[3] = clampFinger(vals[3], RING_MAX);
  target[4] = clampFinger(vals[4], PINKY_MAX);
  target[5] = clampWrist(vals[5]);

  return true;
}

//Setup

void setup() {
  servoindex2.attach(2); // Ring
  servoindex3.attach(3); // Wrist
  servoindex4.attach(4); // Index
  servoindex5.attach(5); // Thumb
  servoindex6.attach(6); // Pinky
  servoindex7.attach(7); // Middle

  Serial.begin(BAUDRATE);
  Serial.println("READY_WAITING_CALIB");

  // safe boot pose
  servoindex3.write(WRIST_NEUTRAL);
  servoindex2.write(0);
  servoindex4.write(0);
  servoindex5.write(0);
  servoindex6.write(0);
  servoindex7.write(0);
}

//loop

void loop() {
  static String buffer = "";

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      parseFrame(buffer);
      Serial.println(target[6]);
      buffer = "";
    } else {
      buffer += c;
    }
  }

 // for (int i = 0; i < 6; i++) {
  //Serial.println(target[6]);
  //}
  if (!calibrationDone) return;


  for (int i = 0; i < 6; i++) {
    current[i] = stepLimit(current[i], target[i]);
  }

  // Apply to servos
  servoindex5.write(current[0]); // Thumb
  servoindex4.write(current[1]); // Index
  servoindex7.write(current[2]); // Middle
  servoindex2.write(current[3]); // Ring
  servoindex6.write(current[4]); // Pinky
  servoindex3.write(current[5]); // Wrist

  delay(15);
}


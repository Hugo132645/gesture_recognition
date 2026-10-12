// This is for the Gesture Opcodes over Serial (Mega 2560)
// I will place my pin map here:
// 2 - ring
// 3 - wrist
// 4 - index
// 5 - thumb
// 6 - pinky
// 7 - middle

//The angles can be tweaked if we change the mechanism later

#include <Servo.h>

//Pins
const int PIN_RING = 2;
const int PIN_WRIST = 3;
const int PIN_INDEX = 4;
const int PIN_THUMB = 5;
const int PIN_PINKY = 6;
const int PIN_MIDDLE = 7;

//Angles (adjustable)
const int RING_CLOSE = 20;    const int RING_OPEN = 120;
const int INDEX_CLOSE = 120;  const int INDEX_OPEN = 0;
const int THUMB_CLOSE = 110;  const int THUMB_OPEN = 0;
const int PINKY_CLOSE = 120;  const int PINKY_OPEN = 0;
const int MIDDLE_CLOSE = 0;   const int MIDDLE_OPEN = 135;

//These can be changed depending on the angle we want
const int WRIST_NEUT = 135;
const int WRIST_LEFT = 150;
const int WRIST_RIGHT = 90;

// Opcodes
const byte OP_OPEN_PALM = 0x01;
const byte OP_FIST = 0x02;
const byte OP_THUMBS_UP = 0x03;
const byte OP_TWO_FINGERS = 0x04;
const byte OP_POINT_LEFT = 0x05;
const byte OP_POINT_RIGHT = 0x06;

//Servos
Servo sRing, sWrist, sIndex, sThumb, sPinky, sMiddle;

// It's a smooth move helper. Not sure if it's gonna work though. Feel free to comment it.
void moveTo(Servo &sv, int target, int step=2, int ms=5) {
  int cur = sv.read();
  if (cur == target) { sv.write(target); return; }
  int dir = (target > cur) ? 1 : -1;
  for (int a = cur; (dir>0)? a<=target : a>=target; a += dir*step) {
    sv.write(a);
    delay(ms);
  }
  sv.write(target);
}

// Poses
void fingersOpenAll() {
  moveTo(sThumb, THUMB_OPEN);
  moveTo(sIndex, INDEX_OPEN);
  moveTo(sMiddle, MIDDLE_OPEN);
  moveTo(sRing, RING_OPEN);
  moveTo(sPinky, PINKY_OPEN);
}
void fingersCloseAll() {
  moveTo(sThumb,  THUMB_CLOSE);
  moveTo(sIndex,  INDEX_CLOSE);
  moveTo(sMiddle, MIDDLE_CLOSE);
  moveTo(sRing,   RING_CLOSE);
  moveTo(sPinky,  PINKY_CLOSE);
}

void setup() {
  Serial.begin(115200);
  sRing.attach(PIN_RING);
  sWrist.attach(PIN_WRIST);
  sIndex.attach(PIN_INDEX);
  sThumb.attach(PIN_THUMB);
  sPinky.attach(PIN_PINKY);
  sMiddle.attach(PIN_MIDDLE);

  //Safety start state (SSS lmao)
  fingersOpenAll();
  sWrist.write(WRIST_NEUT);

  Serial.println("HAND: ready");
}

void loop() {
  if (!Serial.available()) return;
  int v = Serial.read()
  if (v < 0) return;
  byte op = (byte)v;

  switch (op) {
    case OP_OPEN_PALM:
      fingersOpenAll();
      moveTo(sWrist, WRIST_NEUT);
      Serial.println("OP: OPEN_PALM");
      break;
    
    case OP_FIST:
      fingersCloseAll();
      moveTo(sWrist, WRIST_NEUT);
      Serial.println("OP: FIST");
      break;

    case OP_THUMBS_UP:
      moveTo(sThumb,  THUMB_OPEN);
      moveTo(sIndex,  INDEX_CLOSE);
      moveTo(sMiddle, MIDDLE_CLOSE);
      moveTo(sRing,   RING_CLOSE);
      moveTo(sPinky,  PINKY_CLOSE);
      moveTo(sWrist,  WRIST_NEUT);
      Serial.println("OP: THUMBS_UP");
      break;

    case OP_TWO_FINGERS:
      moveTo(sThumb,  THUMB_CLOSE);
      moveTo(sIndex,  INDEX_OPEN);
      moveTo(sMiddle, MIDDLE_OPEN);
      moveTo(sRing,   RING_CLOSE);
      moveTo(sPinky,  PINKY_CLOSE);
      moveTo(sWrist,  WRIST_NEUT);
      Serial.println("OP: TWO_FINGERS");
      break;

    case OP_POINT_LEFT:
      moveTo(sWrist, WRIST_LEFT);
      Serial.println("OP: POINT_LEFT");
      break;

    case OP_POINT_RIGHT:
      moveTo(sWrist, WRIST_RIGHT);
      Serial.println("OP: POINT_RIGHT");
      break;
    
    default:
      break;
  }
}

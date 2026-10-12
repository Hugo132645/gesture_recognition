// ---------------------------------------------------------------------------
// PROJECT: Rock-Paper-Scissors Robotic Arm Controller
// AUTHOR: Ilia
// DESCRIPTION:
// This sketch listens for serial commands from a Python script (TCN Model)
// and executes one of three gestures: Rock, Paper, or Scissors.
// ---------------------------------------------------------------------------

#include <Servo.h>

// ---------------------------------------------------------------------------
// PIN MAPPING
// ---------------------------------------------------------------------------
// Based on the hardware configuration provided:
const int PIN_RING   = 2;
const int PIN_WRIST  = 3;
const int PIN_INDEX  = 4;
const int PIN_THUMB  = 5;
const int PIN_PINKY  = 6;
const int PIN_MIDDLE = 7;

// ---------------------------------------------------------------------------
// SERVO CALIBRATION (ANGLES)
// ---------------------------------------------------------------------------
// These values define the mechanical limits for "Open" and "Close" states.
const int RING_CLOSE = 20;    const int RING_OPEN = 120;
const int INDEX_CLOSE = 120;  const int INDEX_OPEN = 0;
const int THUMB_CLOSE = 110;  const int THUMB_OPEN = 0;
const int PINKY_CLOSE = 120;  const int PINKY_OPEN = 0;
const int MIDDLE_CLOSE = 0;   const int MIDDLE_OPEN = 135;

// Wrist positions
const int WRIST_NEUT = 135;

// ---------------------------------------------------------------------------
// COMMUNICATION OPCODES
// ---------------------------------------------------------------------------
// Send these bytes from Python to trigger specific moves.
const byte OP_READY    = 0x00; // Reset to neutral state
const byte OP_ROCK     = 0x01; // Fist
const byte OP_PAPER    = 0x02; // Open Palm
const byte OP_SCISSORS = 0x03; // Index + Middle fingers extended

// ---------------------------------------------------------------------------
// GLOBALS
// ---------------------------------------------------------------------------
Servo sRing, sWrist, sIndex, sThumb, sPinky, sMiddle;

// ---------------------------------------------------------------------------
// HELPER FUNCTIONS
// ---------------------------------------------------------------------------

/**
 * moveTo:
 * Moves a servo smoothly from its current position to the target position.
 * @param sv     Reference to the Servo object
 * @param target Target angle (0-180)
 * @param step   Angle increment per iteration (controls smoothness)
 * @param ms     Delay in milliseconds per step (controls speed)
 */
void moveTo(Servo &sv, int target, int step=2, int ms=5) {
  int cur = sv.read();
  if (cur == target) {
    sv.write(target);
    return;
  }
  
  int dir = (target > cur) ? 1 : -1;
  
  // Incremental movement loop
  for (int a = cur; (dir > 0) ? a <= target : a >= target; a += dir * step) {
    sv.write(a);
    delay(ms);
  }
  // Ensure final position is exact
  sv.write(target);
}

// ---------------------------------------------------------------------------
// GESTURE DEFINITIONS
// ---------------------------------------------------------------------------

// Gesture: PAPER (Open Palm)
void makePaper() {
  moveTo(sThumb,  THUMB_OPEN);
  moveTo(sIndex,  INDEX_OPEN);
  moveTo(sMiddle, MIDDLE_OPEN);
  moveTo(sRing,   RING_OPEN);
  moveTo(sPinky,  PINKY_OPEN);
}

// Gesture: ROCK (Fist)
void makeRock() {
  moveTo(sThumb,  THUMB_CLOSE);
  moveTo(sIndex,  INDEX_CLOSE);
  moveTo(sMiddle, MIDDLE_CLOSE);
  moveTo(sRing,   RING_CLOSE);
  moveTo(sPinky,  PINKY_CLOSE);
}

// Gesture: SCISSORS (V-Sign)
void makeScissors() {
  // Open Index and Middle
  moveTo(sIndex,  INDEX_OPEN);
  moveTo(sMiddle, MIDDLE_OPEN);
  
  // Close others
  moveTo(sThumb,  THUMB_CLOSE);
  moveTo(sRing,   RING_CLOSE);
  moveTo(sPinky,  PINKY_CLOSE);
}

// ---------------------------------------------------------------------------
// MAIN SETUP
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200); // Ensure Python matches this baud rate

  // Attach servos to pins
  sRing.attach(PIN_RING);
  sWrist.attach(PIN_WRIST);
  sIndex.attach(PIN_INDEX);
  sThumb.attach(PIN_THUMB);
  sPinky.attach(PIN_PINKY);
  sMiddle.attach(PIN_MIDDLE);

  // Initial State: Ready (Paper/Open is usually the safest start position)
  makePaper();
  sWrist.write(WRIST_NEUT);

  Serial.println("SYSTEM: RPS Controller Ready");
}

// ---------------------------------------------------------------------------
// MAIN LOOP
// ---------------------------------------------------------------------------
void loop() {
  // Wait for incoming data
  if (!Serial.available()) return;

  // Read the opcode byte
  int v = Serial.read();
  if (v < 0) return; // Safety check
  byte op = (byte)v;

  switch (op) {
    case OP_READY:
      makePaper();
      Serial.println("ACTION: READY_STATE");
      break;

    case OP_ROCK:
      makeRock();
      Serial.println("ACTION: ROCK");
      break;

    case OP_PAPER:
      makePaper();
      Serial.println("ACTION: PAPER");
      break;

    case OP_SCISSORS:
      makeScissors();
      Serial.println("ACTION: SCISSORS");
      break;
      
    default:
      // Ignore unknown opcodes or implement an error LED blink here
      break;
  }
}
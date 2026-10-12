#include <Servo.h>

// Pins (Same as before)
const int PIN_RING = 2;
const int PIN_WRIST = 3;
const int PIN_INDEX = 4;
const int PIN_THUMB = 5;
const int PIN_PINKY = 6;
const int PIN_MIDDLE = 7;
const int PIN_WRIST_PITCH = 8; // EDIT
// Servos (Same as before)
Servo sRing, sWrist, sIndex, sThumb, sPinky, sMiddle;

void setup() {
  Serial.begin(115200);
  sRing.attach(PIN_RING);
  sWrist.attach(PIN_WRIST);
  sIndex.attach(PIN_INDEX);
  sThumb.attach(PIN_THUMB);
  sPinky.attach(PIN_PINKY);
  sMiddle.attach(PIN_MIDDLE);
  sWristPitch.attach(PIN_WRIST_PITCH);

  // Set a safe starting position
  sThumb.write(0);
  sIndex.write(0);
  sMiddle.write(135);
  sRing.write(120);
  sPinky.write(0);
  sWrist.write(135); // WRIST_NEUT
  sWristPitch.write(135); // EDIT 
  Serial.println("HAND: Proportional receiver ready.");
}

void loop() {
  // Check if data is available
  if (Serial.available() > 0) {
    // Read a full string from serial, until the newline '\n'
    String input = Serial.readStringUntil('\n');

    // 1. Check if it's a valid data packet (starts with '<', ends with '>')
    if (input.startsWith("<") && input.endsWith(">")) {
      
      // Remove the '<' and '>' markers
      input = input.substring(1, input.length() - 1);

      // We now have a string like "90,120,0,20,120,90"
      
      // 2. Parse the string to get the angles
      // We will read them in the order Python sent them.

      int angles[7]; // Array to hold the 6 angles
      int i = 0;
      
      char* str = (char*)input.c_str(); // Convert String to char* for strtok
      char* token = strtok(str, ",");   // Split string by commas

      while (token != NULL && i < 7) {
        angles[i] = atoi(token); // Convert token (text) to integer
        token = strtok(NULL, ",");
        i++;
      }

      // 3. If we successfully read 6 angles, move the servos
      if (i == 7) {
        // Order: <thumb, index, middle, ring, pinky, wrist>
        sThumb.write(angles[0]);
        sIndex.write(angles[1]);
        sMiddle.write(angles[2]);
        sRing.write(angles[3]);
        sPinky.write(angles[4]);
        sWrist.write(angles[5]);
        sWristPitch.write(angles[6]);
        // Optional: Send a confirmation back to Python
        // Serial.println("OK"); 
      } else {
        Serial.println("Error: Malformed packet.");
      }
    }
  }
}

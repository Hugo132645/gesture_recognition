#include <Servo.h>



Servo servoindex,servoindex4, servoindex5, servoindex6, servoindex7,servoindex2, servoindex3;         // Define index servo

void setup() { 
//servoindex.attach(22);  // Set index servo to digital pin 
//servoindex12.attach(12);  // Set index servo to digital pin 12
//servoindex13.attach(13);  // Set index servo to digital pin 13

servoindex2.attach(2);  // Set index servo to digital pin 2
servoindex3.attach(3);  // Set index servo to digital pin 3
servoindex4.attach(4);  // Set index servo to digital pin 4
servoindex5.attach(5);  // Set index servo to digital pin 5
servoindex6.attach(6);  // Set index servo to digital pin 6
servoindex7.attach(7); 

Serial.begin(115200);
Serial.println("Start");
} 


void loop() {            // Loop through motion tests

servoindex3.write(135);// 0 is 135 wrist (rotation of -15 in between 0 and 150)
servoindex2.write(0); //20 maximum (ring finger)

servoindex4.write(0); //120 maximum (index finger)
servoindex5.write(0); //110 maximum (thumb)
servoindex6.write(0); //120 maximum (pinky)
servoindex7.write(0); //0 maximum (middle finger)
delay(3000);

servoindex3.write(135);// 0 is 135 wrist
servoindex2.write(0);

servoindex4.write(0);
servoindex5.write(0);
servoindex6.write(0);
servoindex7.write(0);
delay(3000);
}


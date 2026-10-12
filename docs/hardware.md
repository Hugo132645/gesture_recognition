# Hardware and serial protocol

The integrated sketch targets an Arduino Mega with five finger servos and one wrist servo. Use a power supply appropriate for the servos and a common ground with the Arduino; verify the mechanical ranges on your own assembly before applying motion commands.

| Joint | Digital pin | Firmware range (degrees) |
| --- | --- | --- |
| Thumb | 5 | 0–100 |
| Index | 4 | 0–110 |
| Middle | 7 | 0–90 |
| Ring | 2 | 0–110 |
| Pinky | 6 | 0–100 |
| Wrist | 3 | 0–180; boot position 135 |

Serial uses 115200 baud by default. Commands are ASCII, enclosed in angle brackets, and terminated by a newline:

```text
<THUMB,INDEX,MIDDLE,RING,PINKY,WRIST,WRIST_PITCH>
```

Values are integer degrees. The firmware requires seven values but only applies the first six. Finger limits and a maximum step of four degrees per 15 ms loop are enforced in the sketch.

The firmware starts in an open-hand pose and prints `READY_WAITING_CALIB`. It accepts `<CALIB_DONE>\n`, replies `ACK_CALIB`, and then enables motion. The Python script currently sends angle frames but not this enabling message: this is an existing integration gap, retained during repository reorganization.

`firmware/testing_script/testing_script.ino` is a standalone motor motion test, not the live tracking firmware. It actively moves motors when run.

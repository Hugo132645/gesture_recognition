import csv
import serial
import time

# --- CONFIG ---
SERIAL_PORT = 'COM3'  # <-- Change this to your Arduino's port
BAUD_RATE = 115200
CSV_FILE = 'hand_log_quantized.csv'
# -------------

# Connect to the Arduino
print(f"Connecting to {SERIAL_PORT}...")
try:
    # The timeout is important so we don't block forever
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Give Arduino time to reset
    print("Connected.")
except serial.SerialException as e:
    print(f"Error: Could not open port {SERIAL_PORT}. {e}")
    exit()

print(f"Opening CSV file: {CSV_FILE}")

try:
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        
        # Read the column headers from the CSV
        # (e.g., 'servo_thumb_deg*', 'servo_index_deg*', etc.)
        headers = reader.fieldnames
        
        print("CSV opened. Starting playback...")
        
        last_time = time.time()

        for row in reader:
            # 1. Get the angles from the CSV row
            # We need to make sure we get them in a consistent order
            try:
                thumb = float(row['servo_thumb_deg*'])
                index = float(row['servo_index_deg*'])
                middle = float(row['servo_middle_deg*'])
                ring = float(row['servo_ring_deg*'])
                pinky = float(row['servo_pinky_deg*'])
                
                # Read the raw wrist roll from the CSV
                # This value is typically from -90 to +90
                raw_roll = float(row['wrist_roll_deg*'])
                
                # Map the roll to a servo angle (0-180)
                # We'll map roll=0 to servo=90 (neutral)
                # roll=-90 to servo=0
                # roll=+90 to servo=180
                wrist = 90.0 + raw_roll
                
                # Clamp the value to the servo's 0-180 range
                wrist = max(0.0, min(180.0, wrist))
                
            except (ValueError, TypeError):
                # Skip rows with empty or bad data
                print("Skipping bad row...")
                continue

            # 2. Format them into a string
            # We use '<' and '>' as markers.
            # Format: <thumb,index,middle,ring,pinky,wrist>
            data_string = f"<{thumb:.0f},{index:.0f},{middle:.0f},{ring:.0f},{pinky:.0f},{wrist:.0f}>\n"

            # 3. Send the string to the Arduino
            ser.write(data_string.encode('ascii'))
            
            # Optional: Print what you're sending
            print(f"Sent: {data_string}", end='')

            # 4. Wait for Arduino to respond (optional but good practice)
            # response = ser.readline().decode('ascii').strip()
            # print(f"Arduino said: {response}")
            
            # Control the playback speed
            # The original logger saved data every 0.2s
            # We can play it back at that speed.
            time.sleep(0.2) 

finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
    print("\nPlayback finished. Serial port closed.")
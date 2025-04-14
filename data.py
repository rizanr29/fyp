import gspread
import csv
import numpy as np
import os
import time
from google.oauth2.service_account import Credentials
from utils_mp import JuggleTracker

# Path to JSON key file (Google Cloud Console)
SERVICE_ACCOUNT_FILE = "service.json"

# Define the required API scopes
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Authenticate and create a Google Sheets client
try:
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    client = gspread.authorize(creds)
    SHEET_ID = "YourID"
    sheet = client.open_by_key(SHEET_ID).sheet1  # Access the first sheet
    print("✅ Google Sheets connection established.")
except Exception as e:
    print(f"❌ Failed to connect to Google Sheets: {e}")

class JugglingSession:
    def __init__(self, csv_filename="juggling_data.csv"):
        self.total_juggles = 0
        self.juggle_timestamps = []  
        self.ball_heights = []  
        self.first_juggle_time = None
        self.last_juggle_time = None
        self.csv_filename = os.path.abspath(csv_filename)  # Use absolute path
        self.ensure_csv_exists()
        self.improvement_score = 0
        self.endurance_score = 0

        # Initialize a JuggleTracker instance
        self.juggle_tracker = JuggleTracker()

    def ensure_csv_exists(self):
        """ Creates the CSV file if it does not exist to prevent errors. """
        if not os.path.isfile(self.csv_filename):
            with open(self.csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Total Juggles", "Avg Time Gap (s)", "Consistency", "Total Time (s)", "Rating"])
            print(f"📂 Created new CSV file: {self.csv_filename}")

    def record_juggle(self, ball_height, ball_x=None):
        """ Records a new juggle event with timestamp, ball height, and optional x position.
            Also detects drops based on time gap between juggles.
        """
        current_time = time.time()

        if self.first_juggle_time is None:
            self.first_juggle_time = current_time  # ✅ Set first juggle time

        # ✅ Detect drop if time gap is too large
        if self.last_juggle_time is not None:
            time_gap = current_time - self.last_juggle_time
            if time_gap > 3:  # 🔥 Customize threshold (e.g., 3 seconds)
                self.juggle_tracker.ball_drops += 1
                print(f"⚠️ Ball drop detected! Total Drops: {self.juggle_tracker.ball_drops}")

        self.total_juggles += 1
        self.juggle_timestamps.append(current_time)
        self.ball_heights.append(ball_height)

        # ✅ Ensure JuggleTracker receives updated values
        self.juggle_tracker.juggle_timestamps.append(current_time)
        self.juggle_tracker.ball_heights.append(ball_height)
        self.juggle_tracker.juggle_count = self.total_juggles  # ✅ Sync count
        self.latest_consistency = 1.

        if ball_x is not None:
            self.juggle_tracker.ball_x_positions.append(ball_x)  # ✅ Store ball_x

        self.last_juggle_time = current_time  # ✅ Update last juggle time
        self.juggle_tracker.last_juggle_time = current_time  # ✅ Sync with JuggleTracker

    def calculate_time_gap(self):
        """ Calculates the average time gap between juggles. """
        if len(self.juggle_tracker.juggle_timestamps) > 1:
            return self.juggle_tracker.calculate_time_gap()
        
        print("⚠️ Not enough timestamps to calculate time gap.")
        return 0

    def get_total_time(self):
        """ Returns the total duration from the first to last juggle. """
        if self.first_juggle_time is None or self.last_juggle_time is None:
            print("⚠️ First or last juggle time missing. Returning 0.")
            return 0  

        return round(self.last_juggle_time - self.first_juggle_time, 2)  # ✅ Ensure rounded output

    def get_consistency(self):
        """ Returns the latest consistency value stored in the class. """
        print(f"[DEBUG] Consistency: {self.latest_consistency}")  # ✅ Corrected print
        return self.latest_consistency  # ✅ Fetch from class attribute

    def calculate_juggling_rating(self):
        """ Computes juggling rating based on the new scoring system. """
        # 1. Juggling Count (JC)
        juggling_count_score = min(self.total_juggles // 10, 20)  # 1 point per 10 juggles, capped at 20

        # 2. Total Time Taken (TTT)
        total_time = self.get_total_time()
        benchmark_time = 120  # Benchmark for 120 seconds
        if total_time == 0:   # Explicitly handle missing data
            time_score = 0
        elif total_time <= benchmark_time:
            time_score = 10 + ((benchmark_time - total_time) // 5)  # Add 1 point per 5 seconds under benchmark
        else:
            time_score = 10 - ((total_time - benchmark_time) // 5)  # Subtract 1 point per 5 seconds over benchmark
        time_score = max(0, min(20, time_score))  # Cap between 0 and 20

        avg_time_gap = self.calculate_time_gap()
        if avg_time_gap == 0:  # Explicitly handle missing data
            time_gap_score = 0
        elif avg_time_gap <= 0.5:
            time_gap_score = 10
        elif avg_time_gap <= 1.0:
            time_gap_score = 5
        elif avg_time_gap <= 2.0:
            time_gap_score = 2
        else:
            time_gap_score = 0

        # 4. Consistency (C)
        consistency_score_raw = self.get_consistency()  # Get raw consistency score (0 to 1
        if self.total_juggles == 0:  # Check if juggle count is zero
            print("[DEBUG] Juggle count is 0. Setting consistency score to 0.")
            consistency_score = 0
        if consistency_score_raw >= 0.9:
            consistency_score = 10
        elif consistency_score_raw >= 0.7:
            consistency_score = 7
        elif consistency_score_raw >= 0.5:
            consistency_score = 5
        elif consistency_score_raw >= 0.3:
            consistency_score = 3
        else:
            consistency_score = 0  # No points for consistency < 0.3

        # 5. Endurance Point (EP)
        self.endurance_score = 0
        if total_time > 120:
            self.endurance_score = min((total_time - 120) // 10, 10)  # 1 point per 10 seconds beyond 120s, capped at 10

      # 6. Improvement Point (IP)
        self.improvement_score = 0
        if len(self.ball_heights) >= 10:
            first_avg = np.mean(self.ball_heights[:5])
            last_avg = np.mean(self.ball_heights[-5:])
            print(f"[DEBUG] BALL HEIGHTS: {self.ball_heights}")
            improvement_percentage = ((last_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0
            # Ensure improvement score is non-negative
            if improvement_percentage > 0:
                self.improvement_score = min(improvement_percentage // 5, 10)  # 1 point per 5% improvement, capped at 10
            else:
                self.improvement_score = 0  # No improvement or negative improvement
            print(f"[DEBUG] Improvement Score: {self.improvement_score}")

        # Total Score Calculation
        total_score = (
            juggling_count_score +
            time_score +
            time_gap_score +
            consistency_score +
            self.endurance_score +
            self.improvement_score
        )

        # Debugging Logs
        print(f"[DEBUG] Juggling Count Score: {juggling_count_score}")
        print(f"[DEBUG] Time Score: {time_score}")
        print(f"[DEBUG] Time Gap Score: {time_gap_score}")
        print(f"[DEBUG] Consistency Score (Raw): {consistency_score_raw}")
        print(f"[DEBUG] Consistency Score (Mapped): {consistency_score}")
        print(f"[DEBUG] Endurance Score: {self.endurance_score}")
        print(f"[DEBUG] Improvement Score: {self.improvement_score}")
        print(f"[DEBUG] Total Score: {total_score}")

        return total_score

    def save_to_csv(self, avg_time_gap, consistency, total_time):
        """ Saves session data to a CSV file. """
        file_exists = os.path.isfile(self.csv_filename)
        headers = ["Total Juggles", "Avg Time Gap (s)", "Consistency", "Total Time (s)", "Endurance", "Improvement", "Rating"]

        try:
            with open(self.csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)

                if not file_exists:
                    writer.writerow(headers)  # Write header if new file

                writer.writerow([
                    self.total_juggles,
                    avg_time_gap,
                    consistency,
                    total_time,
                    self.endurance_score,  # Use class attribute
                    self.improvement_score,  # Use class attribute
                    self.calculate_juggling_rating()
                ])

            print(f"✅ Session saved to {self.csv_filename}")

        except Exception as e:
            print(f"❌ Error saving to CSV: {e}")

    
    '''def save_to_csv(self, juggle_tracker, filename="juggling_data.csv"):
        """ Saves session data to a CSV file after recalculating metrics to ensure accurate values. """
        file_exists = os.path.isfile(filename)
        headers = ["Total Juggles", "Avg Time Gap (s)", "Consistency", "Total Time (s)", "Rating"]

        # ✅ Fetch latest calculated values from JuggleTracker
        avg_time_gap = self.juggle_tracker.calculate_time_gap()
        consistency = self.juggle_tracker.calculate_consistency()  # 🔥 Now using juggle_tracker directly
        total_time = self.juggle_tracker.get_total_time()
        rating = self.calculate_juggling_rating()

        # ✅ Debug print before writing to CSV
        print(f"[DEBUG] Saving to CSV - Juggles: {juggle_tracker.juggle_count}, Time Gap: {avg_time_gap}, Consistency: {consistency}, Total Time: {total_time}, Rating: {rating}")

        try:
            with open(filename, mode="a", newline="") as file:
                writer = csv.writer(file)

                if not file_exists:
                    writer.writerow(headers)  # Write header if new file

                writer.writerow([
                    juggle_tracker.juggle_count,  
                    avg_time_gap,
                    consistency,
                    total_time,
                    rating
                ])

            print(f"✅ Session saved to {filename} with updated metrics.")

        except Exception as e:
            print(f"❌ Error saving to CSV: {e}")'''

    def reset_session(self):
        """ Resets all juggling data for a new session. """
        self.total_juggles = 0
        self.juggle_timestamps.clear()
        self.ball_heights.clear()
        self.juggle_tracker.ball_x_positions.clear()
        self.juggle_tracker.ball_heights.clear()
        self.first_juggle_time = None
        self.last_juggle_time = None
        print("🔄 Session reset.")

    def save_ball_positions(self, ball_x, ball_y):
        """ Saves the ball's (x, y) position after each successful juggle to a local CSV file. """
        try:
            with open("ball_positions.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self.total_juggles, ball_x, ball_y])
            print(f"✅ Ball position saved: Juggle {self.total_juggles}, X: {ball_x}, Y: {ball_y}")
        except PermissionError:
            print(f"❌ Permission denied: Unable to write to 'ball_positions.csv'. Ensure the file is not open in another program.")
        except Exception as e:
            print(f"❌ Error saving ball position: {e}")

def update_google_sheet():
    """ Uploads the latest data from the CSV file to Google Sheets. """
    try:
        with open("juggling_data.csv", mode="r", newline="") as file:
            reader = csv.reader(file)
            rows = list(reader)

            if len(rows) < 2:
                print("⚠️ No valid data to upload.")
                return

            headers = rows[0]  # First row contains column names
            latest_entry = rows[-1]  # Last row contains the latest recorded session
            
            # Check existing sheet data
            existing_data = sheet.get_all_values()

            # Upload headers if the sheet is empty
            if len(existing_data) == 0:
                sheet.append_row(headers)
            
            # Append the latest data
            sheet.append_row(latest_entry)
            print("✅ Google Sheet updated successfully.")

    except Exception as e:
        print(f"❌ Error updating Google Sheet: {e}")
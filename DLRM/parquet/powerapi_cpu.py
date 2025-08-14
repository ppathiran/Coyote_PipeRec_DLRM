import subprocess
import threading
import time
from powerapi import Sensor, Processor, Report
from powerapi.sensor import RAPL
import psutil

def monitor_power(script_pid):
    """Monitor power usage for the given PID."""
    rapl_sensor = RAPL(interval=1000, process=script_pid)
    processor = Processor()
    report = Report()
    rapl_sensor.add_report(report)
    rapl_sensor.attach_processor(processor)
    rapl_sensor.start()

# Run the target script as a subprocess
script_command = ["python", "your_script.py", "--n-jobs", "8", "--modulus", "8192"]
process = subprocess.Popen(script_command)

# Monitor the power of the script
time.sleep(1)  # Wait for the script to initialize
monitor_thread = threading.Thread(target=monitor_power, args=(process.pid,))
monitor_thread.start()

# Wait for the script to complete
process.wait()
print("Script execution completed.")

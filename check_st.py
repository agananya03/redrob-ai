import subprocess
import time
p = subprocess.Popen(["streamlit", "run", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
time.sleep(3)
p.terminate()
stdout, stderr = p.communicate()
print("STDOUT:", stdout)
print("STDERR:", stderr)

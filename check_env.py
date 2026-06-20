import os
import glob

files = glob.glob('.env*')
for f in files:
    try:
        mtime = os.path.getmtime(f)
        with open(f, 'r') as file:
            content = file.read().strip()
        if "=" in content:
            key, val = content.split("=", 1)
            masked = val[:5] + "***" + val[-3:] if len(val) > 8 else val
            print(f"{f}: {key}={masked} (Length: {len(val)})")
        else:
            print(f"{f}: (No = found) content length {len(content)}")
    except Exception as e:
        print(f"Error reading {f}: {e}")

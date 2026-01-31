from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import time
from pathlib import Path

WATCH_DIR = Path("/app")
DEBOUNCE_SECONDS = 1.0

class Handler(FileSystemEventHandler):
    def __init__(self):
        self.last_run = 0

    def on_modified(self, event):
        if event.is_directory:
            return

        # Ignore temp / editor files
        if event.src_path.endswith((".swp", ".tmp", "~")):
            return

        now = time.time()
        if now - self.last_run < DEBOUNCE_SECONDS:
            return

        self.last_run = now
        print("File saved, running process_and_upload.py")
        subprocess.run(["python", "scripts/process_and_upload.py"])

if __name__ == "__main__":
    observer = Observer()
    observer.schedule(Handler(), str(WATCH_DIR), recursive=True)
    observer.start()

    print("Watching for file saves...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

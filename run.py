import subprocess
import time
import os
import sys
import webbrowser

def run():
    print("Starting SVM Digit Classifier...")
    
    # Start backend
    print("Starting Backend (FastAPI on port 8000)...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=os.getcwd()
    )
    
    # Start frontend
    print("Starting Frontend (Vite on port 3000)...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev", "--", "--port", "3000"],
        cwd=os.path.join(os.getcwd(), 'frontend'),
        shell=True # Needed for Windows to find npm
    )
    
    time.sleep(5) # Wait for servers to start
    
    print("All systems go! Local URL: http://localhost:3000")
    webbrowser.open("http://localhost:3000")
    
    try:
        while True:
            if backend_proc.poll() is not None:
                print("Backend stopped.")
                break
            if frontend_proc.poll() is not None:
                print("Frontend stopped.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping servers...")
        backend_proc.terminate()
        frontend_proc.terminate()

if __name__ == "__main__":
    run()

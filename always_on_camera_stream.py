# Web streaming example
# Source code from the official PiCamera package
# http://picamera.readthedocs.io/en/latest/recipes2.html#web-streaming
import RPi.GPIO as GPIO
import io
import time
import picamera
import logging
import socketserver
from threading import Condition
from http import server
import threading
import random
import requests

DETECTEDURL = "http://192.168.1.172:5000/birdDetect"
ENDEDURL = "http://192.168.1.172:5000/birdEnd"


#Motion Detection Setup
GPIO.setmode(GPIO.BOARD)
motionSens = 12
GPIO.setup(motionSens, GPIO.IN)



class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    
    def do_GET(self):
        if self.path == '/open':
            print("FEED THE BIRDS")
            self.send_response(200)
            #self.send_header('Location', '/index.html')
            self.end_headers()

        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            while True:
                with output.condition:
                    output.condition.wait()
                    frame = output.frame
                try:
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
                except Exception as identifier:
                    print(identifier)
                    try:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    except Exception as identifier:
                        print(identifier)

                        

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def stream_video(server):
    server.serve_forever()
    


stream_thread = None
camera = None
motionEvent = False
address = ('', 8000)
server = StreamingServer(address, StreamingHandler)
output = StreamingOutput()
camera = picamera.PiCamera(resolution='250x250', framerate=24)
camera.start_recording(output, format='mjpeg')
stream_thread = threading.Thread(target=stream_video,args=(server,))
stream_thread.start()

while True:
    if not GPIO.input(motionSens) and motionEvent:
        #Motion Event Stops, shut off camera, stop streaming
        print("Motion Event Stop")
        PARAMS = {'motion': True}
       
        motionEvent = False
        #camera.stop_recording()
        #camera.close()
        #server.shutdown()
        #stream_thread.join()
        
        r = requests.get(url = ENDEDURL)
        print(str(r))

    if GPIO.input(motionSens) and motionEvent == False:
        #Motion Event Starts, turn on camera, start recording and stream video
        #in a new thread
        print("Motion Event Start")
        r = requests.get(url = DETECTEDURL)
        print(str(r))
        #time.sleep(1)
        motionEvent = True
        

    


        
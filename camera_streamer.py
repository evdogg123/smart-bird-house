# Web streaming example
# Source code from the official PiCamera package
# http://picamera.readthedocs.io/en/latest/recipes2.html#web-streaming
import RPi.GPIO as GPIO
import io
import picamera
import logging
import socketserver
from threading import Condition
from http import server
import threading
import random

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
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            num = random.randint(0,20)
            while True:
                print(num)

                with output.condition:
                    output.condition.wait()
                    frame = output.frame
                self.wfile.write(b'--FRAME\r\n')
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', len(frame))
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.write(b'\r\n')
                    

'''
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
                raise Exception("CHRISTIFAH IM WALKING HERE 2")
     

        else:
            self.send_error(404)
            self.end_headers()

'''

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

while True:
    if not GPIO.input(motionSens) and motionEvent:
        print("Motion Event Stop")
        motionEvent = False
        print("yo")
        camera.stop_recording()
        print("yo1")
        camera.close()
        print("yo2")
        server.shutdown()
        print("yo3")
        stream_thread.join()
        #server = StreamingServer(address, StreamingHandler)
        print("yo4")

    if GPIO.input(motionSens) and motionEvent == False:
        print("Motion Event Start")
        motionEvent = True
        camera = picamera.PiCamera(resolution='250x250', framerate=24)
        print("hello")
        camera.start_recording(output, format='mjpeg')
        stream_thread = threading.Thread(target=stream_video,args=(server,))
        
        stream_thread.start()

    


        
        

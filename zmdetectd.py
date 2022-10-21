from zmdetectd import app
import bjoern

if __name__ == "__main__":
    # app.run(host = '0.0.0.0', port = 5002, threaded = False, processes = 1)
    bjoern.run(app, host = '0.0.0.0', port = 5003)
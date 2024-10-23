from CameraWatcher import app

if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host = "0.0.0.0", port = 5003, debug = False, use_reloader = True)

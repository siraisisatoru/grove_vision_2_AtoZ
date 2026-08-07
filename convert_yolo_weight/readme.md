*Note, this process is not working with the default grove vision 2 FW.*
This process was able to produce something which doesn's seem to work.

Reference https://github.com/HimaxWiseEyePlus/YOLO11_on_WE2/tree/main 


```sh
docker build --platform linux/amd64 -t grove_vision_convert .
```

```sh
docker run --platform linux/amd64 \
    -v $(pwd):/app \
    grove_vision_convert \
    python convert_tflite.py
```
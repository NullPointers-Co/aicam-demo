# ppp-demo

![Warning Badge](https://img.shields.io/badge/Warning-red)
建议在docker中配置开发环境，或直接运行容器

[关于容器配置的说明在这里](#在容器中运行)

## 环境配置

安装python第三方包

```bash
poetry install
```

安装yolo模型，放在项目目录下的weights中

```bash
mkdir weights  \
    && curl -L -o weights/yolov11n.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
```

## 运行

测试

```bash
python detect.py
```

命令行调用

```bash
Usage: cmdlt.py [OPTIONS]

Options:
  -t, --target PATH       Target file to process
  -f, --reference PATH    Reference file for comparison
  -c, --confidence FLOAT  Confidence threshold for object detection
  --help                  Show this message and exit.
```

## 在容器中运行

下载容器

`docker pull kr1stian/ppp-demo-docker:latest`

测试

`docker run --rm ppp-demo-docker`

自定义target和reference

需要先把target图片和refrence图片放到某一个目录下，然后挂载到容器，最后通过环境变量传参

`docker run --rm -v <your-images-directory>:/mnt -e TARGET="/mnt/<target>" -e REFERENCE="/mnt/<reference>"  ppp-demo-docker`

**注意：your-images-directory, target, reference需要修改成你自己的目录以及文件名称**

最终会生成 d_<your_target_name> 的文件在target同一目录下
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

利用 [model-transform-tool](https://github.com/NullPointers-Co/model-transform-tool) 下载（推荐）

```bash
python transform.py download yolo11x-pose.pt                 
```

从GitHub下载

```bash
mkdir weights  \
    && curl -L -o weights/yolo11n.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
```

## 运行

### 测试

```bash
python detect.py
```

### 命令行调用

```bash
Usage: cmdlt.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  advise
  milling
```

advise

```bash
Usage: cmdlt.py advise [OPTIONS]

Options:
  -t, --target PATH       Target file to process
  -f, --reference PATH    Reference file for comparison
  -c, --confidence FLOAT  Confidence threshold for object detection
  --help                  Show this message and exit.
```

milling

```bash
Usage: cmdlt.py milling [OPTIONS]

Options:
  -t, --type [preview|dataset]  Type of processing  [required]
  -o, --out PATH                Output file path
  -i, --input PATH              Input file path  [required]
  -s, --silent                  Silent mode
  --weight PATH                 Customized weight file path
  --stdio                       output to stdio
  --show-origin                 Preview origin point
  --show-box                    Prevew detetct box
  --help                        Show this message and exit.
```

## 在容器中运行

下载容器

`docker pull kr1stian/ppp-demo-docker:latest`

测试

`docker run --rm ppp-demo-docker`

自定义target和reference

需要先把target图片和refrence图片放到某一个目录下，然后挂载到容器，在命令最后参考[命令行调用](#命令行调用)进行传参

一些🌰

快速测试

```bash
docker run --rm kr1stian/ppp-demo-docker:latest \
  milling -i images/jljt.mp4 \
  --type dataset \
  --stdio
```

用jq解析

```bash
docker run --rm kr1stian/ppp-demo-docker:latest \
  milling -i images/jljt.mp4 \
  --type dataset \
  --stdio --silent | jq '.metadata'
```

运行自定义文件（mount目录）

```bash
docker run --rm -v <your-images-directory>:/mnt \
  kr1stian/ppp-demo-docker:latest \
  milling --help
```

```bash
docker run --rm -v <your-images-directory>:/mnt \
  kr1stian/ppp-demo-docker:latest \
  milling \
  -i <your-images-directory>/<input-file> \
  -o <your-images-directory> \
  -t dataset \
  --stdio \
  --silent
```

用自定义模型生成preview视频

```bash
docker run --rm -v <your-images-directory>:/mnt \
  kr1stian/ppp-demo-docker:latest \
  milling \
  -i <your-images-directory>/<input-file> \
  -o <your-images-directory> \
  --weight <path-to-your-customized-weight> \
  -t preview \
  --show-origin \
  --show-box
```

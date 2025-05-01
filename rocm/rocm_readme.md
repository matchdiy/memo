# ROCm

## 1. Bulid ROCm From Source

### Docker Proxy

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
touch /etc/systemd/system/docker.service.d/http-proxy.conf
```

http-proxy.conf:

```text
[Service]
Environment="HTTP_PROXY=http://proxy.example.com:PORT"
Environment="HTTPS_PROXY=http://proxy.example.com:PORT"
Environment="NO_PROXY=localhost,127.0.0.0/8"
```

重启服务：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### Preinstall

### Build

参考这个文档进行 [ROCm README](https://github.com/ROCm/ROCm/blob/develop/README.md)

* 如果使用docker进行编译，那么preinstall是不需要进行的，docker中都已经安装了。
* `docker run`的命令中需要映射 /etc/group 文件，以便于在docker中使用同主机的用户。
* 无论是docker还是github等海外源，都需要翻墙。可以设置这些应用的代理，但最好不要通过ssh 远程登录到作业机器上进行下载的任务，ssh代理不稳定，资源下载不成功导致各种问题。
* 使用conda构建本地环境，并将其map到docker中，编译需要额外安装
  * `pip install myst_parser`
  * `pip install CppHeaderParser`

## 2. Install ROCm From DEB

使用amdgpu-install进行安装是比通过源码编译安装要方便的多。

___安装 [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/amdgpu-install.html)___

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.4/ubuntu/noble/amdgpu-install_6.4.60400-1_all.deb
sudo apt install ./amdgpu-install_6.4.60400-1_all.deb
sudo apt update
```

___安装驱动___

```bash
amdgpu-install --usecase=dkms
```

___安装ROCm___

```bash
amdgpu-install --usecase=rocm
```

___详细安装选项查询___

```bash
sudo amdgpu-install --list-usecase
```

```text
If --usecase option is not present, the default selection is
"dkms,graphics,opencl,hip"
Available use cases:
dkms            (to only install the kernel mode driver)
  - Kernel mode driver (included in all usecases)
graphics        (for users of graphics applications)
  - Open source Mesa 3D graphics and multimedia libraries
multimedia      (for users of open source multimedia)
  - Open source Mesa 3D multimedia libraries
workstation     (for users of legacy WS applications)
  - Open source multimedia libraries
  - Closed source (legacy) OpenGL
rocm            (for users and developers requiring full ROCm stack)
  - OpenCL (ROCr/KFD based) runtime
  - HIP runtimes
  - Machine learning framework
  - All ROCm libraries and applications
wsl             (for using ROCm in a WSL context)
  - ROCr WSL runtime library (Ubuntu 22.04 only)
rocmdev         (for developers requiring ROCm runtime and
                profiling/debugging tools)
  - HIP runtimes
  - OpenCL runtime
  - Profiler, Tracer and Debugger tools
rocmdevtools    (for developers requiring ROCm profiling/debugging tools)
  - Profiler, Tracer and Debugger tools
amf             (for users of AMF based multimedia)
  - AMF closed source multimedia library
lrt             (for users of applications requiring ROCm runtime)
  - ROCm Compiler and device libraries
  - ROCr runtime and thunk
opencl          (for users of applications requiring OpenCL on Vega or later
                products)
  - ROCr based OpenCL
  - ROCm Language runtime
openclsdk       (for application developers requiring ROCr based OpenCL)
  - ROCr based OpenCL
  - ROCm Language runtime
  - development and SDK files for ROCr based OpenCL
hip             (for users of HIP runtime on AMD products)
  - HIP runtimes
hiplibsdk       (for application developers requiring HIP on AMD products)
  - HIP runtimes
  - ROCm math libraries
  - HIP development libraries
openmpsdk       (for users of openmp/flang on AMD products)
  - OpenMP runtime and devel packages
mllib           (for users executing machine learning workloads)
  - MIOpen hip/tensile libraries
  - Clang OpenCL
  - MIOpen kernels
mlsdk           (for developers executing machine learning workloads)
  - MIOpen development libraries
  - Clang OpenCL development libraries
  - MIOpen kernels
asan            (for users of ASAN enabled ROCm packages)
  - ASAN enabled OpenCL (ROCr/KFD based) runtime
  - ASAN enabled HIP runtimes
  - ASAN enabled Machine learning framework
  - ASAN enabled ROCm libraries
```

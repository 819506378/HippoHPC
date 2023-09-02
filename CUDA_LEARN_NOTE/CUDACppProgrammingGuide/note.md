<span id="picture-logo"></span>

<div> <!--块级封装-->
    <center> <!--将图片和文字居中-->
    <img src="images/Logo_and_CUDA.png"
         style="zoom:200%"/>
    <br> <!--换行-->
    </center>
</div>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"> <<CUDA C++ Programming Guide>>学习笔记</a>

----------------------------------------------

HIPPO
更新日期 20230826
CUDA模块和接口开发文档 V12.2

----------------------------------------------



<span id="Title-1"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction"> 1、介绍</a>

<span id="Title-1.1"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#the-benefits-of-using-gpus"> 1.1、 使用GPU的好处</a>

&emsp;&emsp;GPU(Graphics Processing Unit，图像处理单元)相比于CPU，在使用高指令吞吐量和内存带宽方面有更高的性价比。许多应用充分利用这些高性能特性使得在GPU上跑的比CPU更快。其他的计算硬件，比如FPGAs，也同样的效率，但是编程的灵活性更低。
&emsp;&emsp;因为设计时愿景不同，导致GPU与CPU存在这些性能差异。CPU可以并发执行数十个线程，而GPU的设计可以支持并发上千个线程（分摊较慢的单线程性能以实现更高的吞吐量）。
&emsp;&emsp;GPU专门用于高度并行计算，因此设计了更多的晶体管用于数据处理，而不是数据缓存和流量控制。[图1.1](#picture-1.1)显示了CPU和GPU的芯片资源分布示例

<span id="picture-1.1"></span>

<div> <!--块级封装-->
    <center> <!--将图片和文字居中-->
    <img src="images/gpu-devotes-more-transistors-to-data-processing.png"
         alt="图1.1 GPU将更多的晶体管用于数据处理"
         style="zoom:100%"/>
    <br> <!--换行-->
    <small>图1.1 GPU计算应用。CUDA旨在支持各种语言和应用程序编程接口</small> <!--标题-->
    </center>
</div>

&emsp;&emsp;将更多的晶体管用于数据处理，例如浮点计算，有利于高度并行计算;GPU可以通过计算隐藏内存访问延迟，而不是依赖于大型数据缓存和复杂的流控制来避免长内存访问延迟，这两者在晶体管方面都是昂贵的。
&emsp;&emsp;一般来说，应用程序混合了并行部分和顺序部分，因此系统被设计为混合使用gpu和cpu，以最大限度地提高整体性能。具有高度并行性的应用程序可以利用GPU的这种大规模并行特性来实现比CPU更高的性能。

<span id="Title-1.2"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-a-general-purpose-parallel-computing-platform-and-programming-model"> 1.2、 CUDA®:一个通用并行计算平台和编程模型</a>

&emsp;&emsp;2006年11月，NVIDIA推出了CUDA，一个通用并行计算平台和编程模型，利用NVIDIA gpu中的并行计算引擎，以比CPU更有效的方式解决许多复杂的计算问题。
&emsp;&emsp;CUDA附带了一个软件环境，允许开发人员使用c++作为高级编程语言。如[图1.2](#picture-1.2)所示，支持其他语言、应用程序编程接口或基于指令的方法，例如FORTRAN、DirectCompute、OpenACC。

<span id="picture-1.2"></span>

<div> <!--块级封装-->
    <center> <!--将图片和文字居中-->
    <img src="images/gpu-computing-applications.png"
         alt="无法显示图片时显示的文字"
         style="zoom:100%"/>
    <br> <!--换行-->
    <small>图1.2 GPU计算应用。CUDA旨在支持各种语言和应用程序编程接口</small> <!--标题-->
    </center>
</div>

<span id="Title-1.3"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#a-scalable-programming-model"> 1.3、 CUDA®:一个通用并行计算平台和编程模型</a>

&emsp;&emsp;多核cpu和多核gpu的出现意味着主流处理器芯片现在是并行系统。当前的挑战是开发能够透明地扩展其并行性的应用软件，以利用不断增加的处理器核数，就像3D图形应用程序能够透明地将其并行性扩展到具有不同核数的众核gpu一样。 
&emsp;&emsp;CUDA并行编程模型旨在克服这一挑战，同时对熟悉C等标准编程语言的程序员保持较低的学习曲线。
其核心是三个关键的抽象——线程组的层次结构、共享内存和同步屏障——它们只是作为最小的语言扩展集暴露给程序员。
&emsp;&emsp;这些抽象提供细粒度的数据并行性和线程并行性，嵌套在粗粒度的数据并行性和任务并行性中。它们指导程序员将问题划分为可由线程块独立并行求解的粗粒度子问题，以及可由线程块内所有线程协同并行求解的细粒度子问题。
&emsp;&emsp;这种分解允许线程在解决每个子问题时进行协作，从而保持了语言的表达能力，同时允许自动扩展。 实际上，每个线程块都可以以任意顺序(并发或顺序)在GPU内的任何可用多处理器上调度，因此编译后的CUDA程序可以在任意数量的多处理器上执行，如[图1.3](#picture-1.3)所示，只有运行时系统需要知道物理多处理器数量。
&emsp;&emsp;这种可扩展的编程模型允许GPU架构通过简单地扩展多处理器和内存分区的数量来跨越广泛的市场范围:从高性能爱好者GeForce GPU和专业Quadro和Tesla计算产品到各种廉价的主流GeForce GPU(有关所有cuda支持的GPU的列表，请参阅[支持CUDA的gpu](#Title-6))。

<span id="picture-1.3"></span>

<div> <!--块级封装-->
    <center> <!--将图片和文字居中-->
    <img src="images/automatic-scalability.png"
         alt="无法显示图片时显示的文字"
         style="zoom:200%"/>
    <br> <!--换行-->
    <small>图1.3 自主可伸缩</small> <!--标题-->
    </center>
</div>

> **<center>注意</center>** GPU是围绕一系列流多处理器(Streaming Multiprocessors, SMs)构建的(更多细节请参见[硬件实现](#Title-4))将多线程程序划分为多个独立执行的线程块，使得多处理器数量较多的GPU能够在比多处理器数量较少的GPU更短的时间内自动执行程序。

<span id="Title-1.4"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#document-structure"> 1.4、目录结构</a>
本文档分为以下几个部分:

[1、介绍](#Title-1)是一个CUDA的简介。
[2、编程模型](#Title-2)概述了CUDA编程模型。
[3、编程接口](#Title-3)描述了编程接口。
[4、硬件实现](#Title-4)阐述了硬件实现。
[5、性能指南](#Title-5)给出了一些关于如何达到最大性能的指南。
[6、支持CUDA的GPU](#Title-6)列举了所有CUDA支持的硬件。
[7、C++语言扩展](#Title-7)详细描述了C++语言的扩展。
[8、协作组](#Title-8)描述了不同CUDA线程组的同步单元。
[9、CUDA动态并行](#Title-9)描述了如何发起和同步一个kernel到另一个kernel。
[10、虚拟内存管理](#Title-10)描述了如何管理用以虚拟内存空间。
[11、流有序内存分配器](#Title-11)描述了应用如何申请和释放有序内存。
[12、图像存储节点](#Title-12)描述了图像如何创建和分配自己的内存。
[13、数学方法](#Title-13)列举了CUDA支持的数学方法。
[14、C++语言支持](#Title-14)列举了device端支持的C++特性。
[15、纹理方法](#Title-15)给出了纹理方法更多的细节。
[16、计算能力](#Title-16)给出了各种设备的技术规格，以及更多的架构细节。
[17、Driver API](#Title-17)介绍了低等级的driver API。
[18、CUDA 环境变量](#Title-18)列举了所有的CUDA环境变量。
[19、统一存储编程](#Title-19)介绍了统一内存编程指导。

<small>[[1]](#Title-1.1) : 图形限定符源于这样一个事实:20年前，当GPU最初被创建时，它是作为加速图形渲染的专用处理器设计的。在对实时、高清、3D图形永不满足的市场需求的推动下，它已经发展成为一种通用处理器，用于更多的工作负载，而不仅仅是图形渲染。</small>

<span id="Title-2"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model"> 2、编程模型</a>

本章主要介绍了CUDA编程模型背后的概念，概述了它们是如何暴露在c++中。从[编程接口](#Title-3)的角度对CUDA c++进行了详细的描述。[本章](#Title-2)和[下一章](#Title-3)的vector补充举例可以在<a href="https://docs.nvidia.com/cuda/cuda-samples/index.html#vector-addition"> vectorAdd CUDA sample</a>中找到。

<span id="Title-2.1"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels"> 2.1、Kernels</a>

&emsp;&emsp;CUDA C++扩展了C++，允许程序员自定义C++函数，称为`kernel`，调用时，由N个不同的CUDA线程并行执行N次，而不是像普通的c++函数那样只执行一次。

&emsp;&emsp;`kernel`使用`__global__`声明、使用新符号`<<<...>>>`给定的`kernel`的CUDA线程数量来执行，配置语法(参见 [C++语言扩展](#Title-7))。每个执行`kernel`的线程都有一个唯一的线程ID，在`kernel`中可以通过内置变量访问。

&emsp;&emsp;作为说明，[下面的示例代码](#code-2.1)#使用内置变量`threadIdx`，将两个长度为N的向量A和B相加，并将结果存储到向量C中:

<span id="code-2.1"></span>

``` C++
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```

&emsp;&emsp;在这里，执行VecAdd()的N个线程中的每一个执行一次成对加法。

<span id="Title-2.2"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy"> 2.2、线程层次结构</a>

&emsp;&emsp;为了方便起见，threadIdx是一个3分量向量，因此可以使用一维、二维或三维线程索引来标识线程，形成一维、二维或三维线程块，称为线程`block`。这为跨域元素(如向量、矩阵或体积)调用计算提供了一种自然的方式。

&emsp;&emsp;线程的索引和线程ID以一种直接的方式相互关联:对于一维的`block`，它们是相同的;对于大小为(Dx, Dy)的二维`block`，索引为(x, y)的线程的线程ID为(x + y\*Dx);对于大小为(Dx, Dy, Dz)的三维`block`，索引为(x, y, z)的线程ID为(x + y\*Dx + z\*Dx\*Dy)。

&emsp;&emsp;[示例代码](#code-2.2)将大小为NxN的两个矩阵A和B相加，并将结果存储到矩阵C中:

<span id="code-2.2"></span>

``` C++
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

&emsp;&emsp;每个block的线程数是有限制的，因为一个`block`中的所有线程都应该驻留在同一个多核处理器上，并且必须共享该核有限的内存资源。当前的gpu，线程阻塞可能含有多达1024个线程。

&emsp;&emsp;一个`kernel`可以由多个形状相等的`block`执行，因此线程的总数等于每个线程块的数目乘以`block`的个数。

&emsp;&emsp;如[图2.1](#picture-2.1)所示，线程`block`被组织成一维、二维或三维的线程`block` `grid`。`grid`中`block`的数量通常由正在处理的数据的大小决定，这通常超过了系统中处理器的数量。

<span id="picture-2.1"></span>

<div> <!--块级封装-->
    <center> <!--将图片和文字居中-->
    <img src="images/grid-of-thread-blocks.png"
         alt="无法显示图片时显示的文字"
         style="zoom:100%"/>
    <br> <!--换行-->
    <small>图2.1 Grid线程块</small> <!--标题-->
    </center>
</div>

&emsp;&emsp;每个`block`的线程数和每个`grid`的`block`数在`<<<…>>>`中的语法类型可以是int或dim3。可以像上面的例子那样指定二维`block`或`grid`。

&emsp;&emsp;`grid`中的每个`block`都可以通过一个一维、二维或三维的唯一索引标识，在`kernel`中可以通过内置的`blockIdx`变量访问。线程`block`的尺寸在`kernel`中可以通过内置的blockDim变量访问。

&emsp;&emsp;代码扩展了前面的MatAdd()示例以处理多个`block`，如下所示。

<span id="code-2.3"></span>

```C++
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

&emsp;&emsp;线程`block`大小为16x16(256个线程)，虽然在这里是任意大小，但是一种常见的选择。`grid`由足够的`block`创建，与之前一样，每个矩阵元素有一个线程。为简单起见，[本示例](#code-2.3)假设每个维度中每个`grid`的线程数可以被该维度中每个`block`的线程数整除，尽管事实并非如此。

&emsp;&emsp;线程`block`需要独立执行：必须能够以任意顺序执行它们，并行或串行。这种独立性要求允许线程`block`在任意数量的核心上按任意顺序进行调度，如[图1.3](#picture-1.3)所示，使程序员能够编写与核数量相关的代码。

&emsp;&emsp;同一个`block`中的线程可以通过共享内存共享数据，并通过同步它们的执行来协调内存访问。更准确地说，可以在内核中调用`__syncthreads()`内部函数来指定同步的地方。`__syncthreads()`就像一个屏障，`block`中的所有线程都必须等待，然后才允许继续执行。[共享内存](#Title-3.2.4)给出了一个使用共享内存的例子。除了`__syncthreads()`之外，[协作组API](#Title-8)还提供了一组丰富的线程同步原语。

&emsp;&emsp;为了高效合作，共享内存应该是靠近每个处理器核心的低延迟内存(很像L1缓存)，`__syncthreads()`应该是轻量级的。

<span id="Title-2.2.1"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters"> 2.2.1、线程block cluster</a>

&emsp;&emsp;随着NVIDIA[计算能力9.0](#Title-16.8)的推出，CUDA编程模型引入了一个可选的层次结构级别，称为线程`block`集群，由线程`block`组成。类似于线程`block`保证线程在多核流处理器上的共同调度，`cluster`中的线程`blocks`也保证在GPU上的共同调度。

&emsp;&emsp;与线程块类似，集群也被组织成一维、二维或三维，如[图2.2](#picture-2.2)所示。

<span id="picture-2.2"></span>

<div> <!--块级封装-->
    <center> <!--将图片和文字居中-->
    <img src="images/grid-of-thread-blocks.png"
         alt="无法显示图片时显示的文字"
         style="zoom:100%"/>
    <br> <!--换行-->
    <small>图2.2 自主可伸缩</small> <!--标题-->
    </center>
</div>

&emsp;&emsp;`cluster`中的线程`block`数量可以自定义，在CUDA中最多支持8个线程`block`作为可移植`cluster`大小。请注意，在GPU硬件或MIG配置太小而无法支持8个多处理器的情况下，最大`cluster`大小将相应减少。识别这些较小的配置，以及支持超过8的线程`block` `cluster`大小的较大配置，是特定于体系结构的，可以使用`cudaOccupancyMaxPotentialClusterSize`API进行查询。

> **注意**
> 在使用`cluster`支持启动的`kernel`中，出于兼容性的考虑，gridDim变量仍然以线程`block`的数量表示大小。可以使用cluster Group API找到`cluster`中块的下标。

&emsp;&emsp;线程`block` `cluster`可以在`kernel`中启用，使用`__cluster_dims__(X,Y,Z)`来使用编译器时间`kernel`属性，或使用CUDA`kernel`启动`cudaLaunchKernelEx`。[示例代码](#code-2.4)展示了如何使用编译器时间内核属性启动`cluster`。使用kernel属性的`cluster`大小在编译时是固定的，然后`kernel`可以使用经典的`<<<...>>>`。如果`kernel`使用编译时`cluster`大小，则在启动`kernel`时不能修改`cluster`大小。

<span id="code-2.4"></span>

```C++
// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```

线程`block` `cluster`大小也可以在运行时设置，`kernel`可以使用CUDA`kernel`启动API启动`cudaLaunchKernelEx`。[示例代码](#code-2.5)展示了如何使用可扩展API启动`cluster` `kernel`。

<span id="code-2.5"></span>

```C++
// Kernel definition
// No compile time attribute attached to the kernel
__global__ void cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Kernel invocation with runtime cluster size
    {
        cudaLaunchConfig_t config = {0};
        // The grid dimension is not affected by cluster launch, and is still enumerated
        // using number of blocks.
        // The grid dimension should be a multiple of cluster size.
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, cluster_kernel, input, output);
    }
}
```

&emsp;&emsp;支持计算能力为9.0的gpu，`cluster`中的所有线程`block`都保证在单个GPU处理集群(GPU Processing Cluster, GPC)上共同调度，并允许`cluster`中的线程`block`使用集群组API cluster.sync()执行硬件支持的同步。`cluster`组还提供了成员函数，分别使用`num_threads()`API查询线程数大小，`num_blocks()`API查询线程`block`大小。可以使用`dim_threads()`API查询`cluster group`中线程的下标，使用`dim_blocks()`API查询`cluster group`中`block`的下标。

&emsp;&emsp;属于`cluster`的线程`block`可以访问分布式共享内存。`cluster`中的线程`block`能够对分布式共享内存中的任何地址进行读、写和执行原子操作。[分布式共享](#Title-3.2.5)内存给出了一个在分布式共享内存中执行矩阵的示例。

<span id="Title-2.3"></span>
## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory"> 2.3、内存层级</a>

&emsp;&emsp;CUDA线程可以在执行期间访问多个内存空间中的数据，[如图](#picture-2.3)所示。每个线程都有私有的本地内存。每个线程`block`都有共享内存，对该`block`的所有线程可见，并且与该`block`具有相同的生存期。线程`block cluster`中的线程`block`可以对彼此的共享内存执行读、写和原子操作。所有线程都可以访问相同的全局内存。

&emsp;&emsp;还有两个额外的只读内存空间可供所有线程访问：`constant`和`texture`内存空间。`global`、`constant`和`texture`内存空间针对不同用途的内存进行了优化(参见[设备内存访问](#Title-5.3.2))。`texture`内存还提供了不同的寻址方式，以及一些特定数据格式的数据过滤(参见[`texture`和`surface`内存](#Title-3.2.14))。

&emsp;&emsp;`global`、`constant`和`texture`内存空间始终贯穿应用启动的`kernel`。

<span id="picture-2.3"></span>

<div> <!--块级封装-->
    <center> <!--将图片和文字居中-->
    <img src="images/memory-hierarchy.png"
         alt="无法显示图片时显示的文字"
         style="zoom:100%"/>
    <br> <!--换行-->
    <small>图2.3 内存层级</small> <!--标题-->
    </center>
</div>

<span id="Title-2.4"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#heterogeneous-programming"> 2.4、异构编程</a>

&emsp;&emsp;[如图](#picture-2.4)所示，CUDA编程模型假设CUDA线程在物理上独立的设备上执行，该设备作为运行c++程序`host`的协同处理器运行。例如，当`kernels`在GPU上执行而c++程序的其余部分在CPU上执行时，就是这种情况。

&emsp;&emsp;CUDA编程模型还假设`host`和`device`在DRAM中保持各自独立的内存空间，分别称为`host memory`和`device memory`。因此，程序通过调用CUDA运行时(在[编程接口](#Title-3)中描述)来管理`kernel`可见的`global`、`constant`和`texture`内存空间。这包括设备内存分配和回收，以及`host memory`和`device memory`之间的数据传输。

&emsp;&emsp;统一内存提供托管内存来连接`host memory`和`device memory`。托管内存作为具有公共地址空间的单个连贯内存映像，系统中的所有cpu和gpu都可以访问。此功能允许超设备内存订阅，并且可以通过消除在`host`和`device`上显式镜像数据的需要，大大简化移植应用程序的任务。有关统一内存的介绍，请参阅[统一内存编程](#Title-19)。


<span id="picture-2.4"></span>

<div> <!--块级封装-->
    <center> <!--将图片和文字居中-->
    <img src="images/heterogeneous-programming.png"
         alt="无法显示图片时显示的文字"
         style="zoom:100%"/>
    <br> <!--换行-->
    <small>图2.4 异构编程</small> <!--标题-->
    </center>
</div>

> **注意**
> 串行代码在主机上执行，并行代码在设备上执行。

<span id="Title-2.5"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-simt-programming-model"> 2.5、异步SIMT编程模型</a>

&emsp;&emsp;在CUDA编程模型中，线程是执行计算或内存操作的最低抽象级别。从基于NVIDIA Ampere GPU架构的设备开始，CUDA编程模型通过异步编程模型为内存操作提供加速。

&emsp;&emsp;异步编程模型为CUDA线程之间的同步定义了[异步屏障](#Title-7.26)的行为。该模型还解释并定义了[cuda::memcpy_async](#Title-7.27)如何用于在GPU计算时从全局内存中异步移动数据。

<span id="Title-2.5.1"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface"> 2.5.1、异步操作</a>

&emsp;&emsp;异步操作被定义为由CUDA线程发起并由另一个线程异步执行的操作。In a well formed program one or more CUDA threads synchronize with the asynchronous operation。发起异步操作的CUDA线程不需要在同步线程中。

&emsp;&emsp;异步线程(as-if线程)总是与发起异步操作的CUDA线程相关联。异步操作使用同步对象来完成同步操作。这样的同步对象可以由用户显式地管理(例如，`cuda::memcpy_async`)或在库中隐式地管理(例如，`cooperative_groups::memcpy_async`)。

&emsp;&emsp;同步对象可以是`cuda::barrier`或者`cuda::pipeline`。这些对象在使用`cuda::pipeline`的[异步屏障](#Title-7.26)和[异步数据副本 using cuda::pipeline](#Title-7.27)中有详细的解释。这些同步对象可以在不同的线程作用域中使用。作用域定义了一组线程，这些线程可以使用同步对象来同步异步操作。下表定义了CUDA C++中可用的线程范围以及可以与每个线程同步的线程。

<span id="Table-2.1"></span>

<table class="table-no-stripes colwidths-given docutils align-default">
<colgroup>
<col style="width: 70%">
<col style="width: 30%">
</colgroup>
<thead>
<tr class="row-odd">
<th class="head"><p>Thread Scope</p></th>
<th class="head"><p>Description</p></th>
</tr>
</thead>
<tbody>
<tr class="row-even">
<td><p><code class="docutils literal notranslate"><span class="pre">cuda::thread_scope::thread_scope_thread</span></code></p></td>
<td><p>只有启动异步操作的CUDA线程同步。</p></td>
</tr>
<tr class="row-odd">
<td><p><code class="docutils literal notranslate"><span class="pre">cuda::thread_scope::thread_scope_block</span></code></p></td>
<td><p>所有或任何CUDA线程在相同的线程块作为启动线程同步。</p></td>
</tr>
<tr class="row-even">
<td><p><code class="docutils literal notranslate"><span class="pre">cuda::thread_scope::thread_scope_device</span></code></p></td>
<td><p>在同一GPU设备中的所有或任何CUDA线程与初始线程同步。</p></td>
</tr>
<tr class="row-odd">
<td><p><code class="docutils literal notranslate"><span class="pre">cuda::thread_scope::thread_scope_system</span></code></p></td>
<td><p>所有或任何CUDA或CPU线程在同一系统作为启动线程同步。</p></td>
</tr>
</tbody>
</table>

&emsp;&emsp;这些线程作用域是在CUDA标准c++库中作为标准c++的扩展实现的。

<span id="Title-2.6"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability"> 2.6、计算能力</a>

&emsp;&emsp;设备的计算能力由版本号表示，有时也称为它的"SM版本"。版本号标识GPU硬件支持的特性，应用程序在运行时用来确定当前GPU上有哪些硬件特性、指令可用。

&emsp;&emsp;计算能力由主修订号X和次修订号Y组成，用x.y表示。

&emsp;&emsp;相同主修订号的设备核心架构相同。主修订号为9，基于`NVIDIA Hopper GPU`架构；8，基于`NVIDIA Ampere GPU`架构；7，基于`Volta`架构；6，基于`Maxwell`架构；5，基于`Maxwell`架构；3，基于`Kepler`架构。

&emsp;&emsp;次要修订号对应于对核心体系结构的增量改进，可能包括新特性。

&emsp;&emsp;`Turing`是用于计算能力为7.5的设备的架构，是基于`Volta`架构的增量更新。

&emsp;&emsp;[支持CUDA的GPU](#Title-6)列举了所有CUDA支持的设备及计算能力。[计算能力](#Title-16)给出了每种计算能力的技术说明。

> __注意__
> 特定GPU的计算能力版本不应与CUDA版本(例如，CUDA 7.5, CUDA 8, CUDA 9)混淆，CUDA版本是CUDA软件平台的版本。CUDA平台被应用程序开发人员用来创建在多代GPU架构上运行的应用程序，包括未来有待发明的GPU架构。虽然CUDA平台的新版本通常通过支持新的图形处理器架构的计算能力版本来增加对新的图形处理器架构的本地支持，但 CUDA 平台的新版本通常也包括独立于硬件生成的软件特性。

&emsp;&emsp;从CUDA 7.0开始，不再支持`Tesla`架构。9.0不再支持`Fermi`架构。

<span id="Title-3"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface"> 3、编程接口</a>

&emsp;&emsp;CUDA C++为熟悉C++编程语言的用户提供了一个简单的路径，可以轻松编写供`device`执行的程序。

&emsp;&emsp;它由C++语言的最小扩展集和运行时库组成。

&emsp;&emsp;在[编程模型](#Title-2)中引入了核心语言扩展。开发人员定义C++函数就像一样定义`kernel`，并在每次调用时使用一些新的语法来指定`grid`和`block`。所有扩展的完整描述可以在[C++语言扩展](#Title-7)中找到。任何包含这些扩展名的源文件都必须用`nvcc`编译，如[使用nvcc编译](#Title-3.1)所述。

&emsp;&emsp;在[CUDA运行](#Title-3.2)时中引入了`runtime`。`runtime`提供了在`host`上执行的C和C++函数来分配和释放设备内存，在`host`内存和`device`内存之间传输数据，管理具有多个`device`的系统等。`runtime`的完整描述可以在CUDA参考手册中找到。

&emsp;&emsp;`runtime`建立在一个较低级别的C API, `CUDA driver` API之上，`runtime`也可以被应用程序访问。`CUDA driver` API通过暴露底层概念提供了额外的控制级别，如CUDA `contexts`(模拟`device`的`host`进程)和CUDA 模块(模拟`device`动态加载库)。大多数应用程序不使用`CUDA driver`API，因为它们不需要这种额外的控制级别，并且在使用`runtime`，`context`和`module`管理是隐式的，从而产生更简洁的代码。由于`runtime`可与`CUDA driver`API互操作，因此大多数需要某些`driver`API特性的应用程序可以默认使用`runtime`API，并且仅在需要时使用`driver`API。`driver`API在[Driver API](#Title-17)中介绍，并在参考手册中进行了详细描述。

<span id="Title-3.1"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-with-nvcc"> 3.1、使用nvcc编译</a>

&emsp;&emsp;`kernels`可以使用CUDA指令集架构编写，称为`PTX`。`PTX`参考手册中有描述。然而，使用高级编程语言(如C++)通常更有效。在这两种情况下，`kernels`都必须通过`nvcc`编译成二进制代码才能在`device`上执行。

&emsp;&emsp;`nvcc`是一个编译器驱动程序，它简化了编译C++或`PTX`代码的过程。`nvcc`提供简单而熟悉的命令行选项，并通过调用实现不同编译阶段的工具集合来执行这些选项。本节概述了`nvcc` `workflow`和命令选项。`nvcc`用户手册有完整描述。

<span id="Title-3.1.1"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-workflow"> 3.1.1、编译`workflow`</a>

<span id="Title-3.1.1.1"></span>

#### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#offline-compilation"> 3.1.1.1、离线编译</a>

&emsp;&emsp;用`nvcc`编译的源文件包括`host`代码(即，在`host`上执行的代码)和`device`代码(即，在`device`上执行的代码)。`nvcc`的`basic workflow`包括从`host`上分离的`device`代码，然后：
* 将`device`代码编译成汇编形式(`PTX` 代码)、二进制形式(`Cubin` 对象) 。
* [Kernels](#Title-2.1)中有介绍([执行配置](#Title-7.35)有更多描述)，`nvcc`通过必要的CUDA `runtime`函数调用中引入的`<<<...>>>`语法， 调用`PTX`代码、`Cubin`对象加载和启动每个已编译的`kernel`。

&emsp;&emsp;修改后的`host`代码要么作为C++代码输出，留给其他工具编译，要么在最后编译阶段通过调用`nvcc`调用`host`编译器直接作为目标代码输出。然后应用可以：
* 要么连接到编译`host`代码(这是最常见的情况)
* 要么忽略修改的`host`代码(如果有的话)，并使用`CUDA drvier`API(参见[Drvier API](#Title-17))来加载和执行`PTX`代码或`cubin`对象。

<span id="Title-3.1.1.2"></span>

#### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#just-in-time-compilation"> 3.1.1.2、即时编译</a>

&emsp;&emsp;应用程序在`runtime`加载的任何`PTX`代码都由`device driver`进一步编译为二进制代码。这就是即时编译。即时编译增加了应用程序的加载时间，但允许应用程序从每个新`device driver`带来的任何新的编译器改进中受益。即时编译也是应用程序在非编译时的`device`上运行的唯一方法。详见[应用兼容性](#Title-3.1.4)。

&emsp;&emsp;当`device driver`为某个应用即时编译某些`PTX`代码时，它会自动缓存生成的二进制代码的副本，以避免在应用的后续调用中重复编译。缓存(计算缓存)在`device driver`升级时自动失效，因此应用可以从编译到`device driver`中的新即时编译器的改进中受益。

&emsp;&emsp;环境变量可用于控制即时编译，如[CUDA 环境变量](#Title-18)所述。

&emsp;&emsp;作为使用`nvcc`编译CUDA C++ `device`代码的替代方案。`NVRTC`可以在`runtime`时将CUDA C++ `device`代码编译`PTX`。`NVRTC`是CUDA C++的`runtime`编译库。更多信息可以在`NVRTC`用户指南中找到。

<span id="Title-3.1.2"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#binary-compatibility"> 3.1.2、二进制兼容性</a>

&emsp;&emsp;二进制文件是一个特殊的架构。`cubin`是使用`-code`编译选项指定目标架构生成的，例如：使用`-code=sm_80`编译具有8.0计算能力的`devices`生成二进制代码。二进制兼容性保证一个次要版本到下一个版本的兼容性，但不能保证从一个小版本到前一个小版本或跨主要版本的二进制兼容性。换句话说，为X.y的计算能力生成的`cubin`对象保证能在拥有X.z计算能力的`devices`上运行，只要z>=y。

> __注意__
> 二进制兼容性仅支持`desktop`。二进制兼容性不支持`Tegra`。此外，`desktop`和`Tegra`之间的二进制兼容性也不支持。

<span id="Title-3.1.3"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ptx-compatibility"> 3.1.3、`PTX`兼容性</a>

&emsp;&emsp;一些`PTX`指令仅支持具有更高计算能力的`device`。例如，`Warp Shuffle Functions`仅支持计算能力5.0及以上的`device`。`-arch`编译选项指定将C++编译为假定的计算能力。因此，例如，包含`warp shuffle`的代码必须使用`-arch=compute 50`(或更高)进行编译。

&emsp;&emsp;为某些特定计算能力而产生的`PTX`代码可以被编译成具有更高或同等计算能力的二进制代码。请注意，从早期`PTX`版本编译的二进制文件可能无法使用某些硬件特性。例如，由为计算能力6.0(`Pascal`)生成的`PTX`编译的具有计算能力7.0(`Volta`)的二进制目标`device`将不使用`Tensor Core`指令，因为这些指令在`Pascal`上不可用。因此，最终的二进制文件的性能可能比使用最新版本的`PTX`生成的二进制文件的性能更差。

<span id="Title-3.1.4"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#application-compatibility"> 3.1.4、应用兼容性</a>

&emsp;&emsp;要在具有特定计算能力的`device`上执行代码，应用程序必须加载与此计算能力兼容的二进制代码或`PTX`代码，如[二进制兼容性](#Title-3.1.2)和[`PTX`兼容性](#Title-3.1.3)中所述。特别是，为了能够在计算能力更强的未来架构上执行代码(目前还不能生成二进制代码)，应用程序必须加载为这些`devices`即时编译的`PTX`代码(见[即时编译](#Title-3.1.1.2))。

&emsp;&emsp;在CUDA C++应用程序中嵌入哪些`PTX`和二进制代码由`-arch`和`-code`编译器选项或`-gencode`编译器选项控制，详见`nvcc`用户手册。例如：

<span id="code-3.1"></span>

```C++
nvcc x.cu
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_60,code=sm_60
        -gencode arch=compute_70,code=\"compute_70,sm_70\"
```

&emsp;&emsp;计算能力5.0和6.0的嵌入式二进制代码兼容性(第一选项或第二选项 `-gencode`)，计算能力7.0的`PTX`兼容性和二进制代码兼容性。

&emsp;&emsp;生成`host`代码，以便在`runtime`自动选择要加载和执行的最合适的代码，在[上述示例](#code-3.1)中，这些代码将是:

* 具有5.0和5.2计算能力的`device`的5.0二进制代码
* 具有6.0和6.1计算能力的`device`的6.0二进制代码
* 具有7.0和7.5计算能力的`device`的7.0二进制代码
* 具有8.0和8.6计算能力的`device`的编译成`runtime`二进制代码`PTX`代码

&emsp;&emsp;`x.cu`文件有一个优化代码的路径，它使用`warp`减少操作。例如，只支持在8.0或更高计算能力的设备。`__CUDA_ARCH__`宏可用于根据计算能力区分各种代码路径。`__CUDA_ARCH__`只定义了`device`代码。例如，当使用`-arch=compute_80`编译时，`__CUDA_ARCH__`等于800。

&emsp;&emsp;使用`driver`API的应用程序必须分离文件来编译代码，并且在`runtime`显示加载和执行最合适的文件。

&emsp;&emsp;`Volta`架构架构引入了[独立线程调度](#Title-16.6.2)，它改变了GPU上线程调度的方式。对于依赖于先前体系结构中[`SIMT`调度](#Title-4.1)的特定行为的代码，[独立线程调度](#Title-16.6.2)可能会改变参与线程的集合，从而导致不正确的结果。为了在实现[独立线程调度](#Title-16.6.2)中详细描述的纠正措施的同时帮助迁移，`Volta`开发人员可以使用编译器选项组合`-arch=compute_60 -code=sm_70`来选择`Pascal`的线程调度。

&emsp;&emsp;`nvcc`手册列举了各种`-arch`,`-code`和`-gencode`编译选项的缩写。例如`-arch=sm_70`是`-arch=compute_70 -code=compute_70,sm_70`(或`-gencode arch=compute_70,code=\"compute_70,sm_70\"`)的缩写。

<span id="Title-3.1.5"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-compatibility"> 3.1.5、C++兼容性</a>

&emsp;&emsp;遵循C++语法规则是编译CUDA源文件的前提。`host`代码完整支持C++。然而，就像[C++语言支持](#Title-14)中描述的`device`代码只完全支持C++的一个子集。

<span id="Title-3.1.6"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#bit-compatibility"> 3.1.6、64位兼容性</a>

&emsp;&emsp;64位版本的`nvcc`以64位模式编译`device`代码(即，指针是64位的)。以64位模式编译的`device`代码仅支持以64位模式编译的`host`代码。

<span id="Title-3.2"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-runtime"> 3.2、CUDA 运行时</a>

&emsp;&emsp;`runtime`是在`cudart`库中实现的，通过`cudart.lib`或者`libcudart.a`静态链接到应用，或通过`cudart.dll`或`libcudart.so`动态链接到应用。对于动态链接来说，需要 cutart.dll 和/或 cutart.so 的应用程序通常将它们作为应用程序安装包的一部分。只有在链接到相同CUDA`runtime`实例的组件之间传递CUDA`runtime`符号的地址才是安全的。

&emsp;&emsp;`runtime`所有的入口都是以`cuda`为前缀的。

&emsp;&emsp;正如在[异构编程](#Title-2.4)中提到的，CUDA编程模型假定一个由`host`和`device`组成的系统，每个`host`和`device`都有自己独立的内存。[`device`内存](#Title-3.2.2)概述了用于管理`device`内存的`runtime`函数。

&emsp;&emsp;[共享内存](#Title-3.2.4)演示了如何使用[线程层次结构](#Title-2.2)中引入的共享内存来最大限度地提高性能。

&emsp;&emsp;[页面锁定`host`内存](#Title-3.2.6)引入页面锁定`host`内存，需要重叠`kernel`执行与`host`内存和`device`内存之间的数据传输。

&emsp;&emsp;[异步并发执行](#Title-3.2.8)描述了用于在系统的不同级别上支持异步并发执行的概念和API。

&emsp;&emsp;[多`device`系统](#Title-3.2.9)演示了如何将编程模型扩展到具有多个`device`连接到同一`host`的系统。

&emsp;&emsp;[错误检查](#Title-3.2.12)描述如何正确检查`runtime`生成的错误。

&emsp;&emsp;[调用堆栈](#Title-3.2.13)提到了用于管理CUDA C++调用堆栈的`runtime`函数。

&emsp;&emsp;[`texture`和`surface`内存](#Title-3.2.14)提供了`texture`和`surface`存储器空间，提供了另一种访问`device`存储器的方式; 它们还公开了GPU`texture`硬件的一个子集。

&emsp;&emsp;[图形互操作性](#Title-3.2.15)引入了`runtime`提供的与两个主要图形API(`OpenGL`和`Direct3D`)互操作的各种功能。

<span id="Title-3.2.1"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#initialization"> 3.2.1、初始化</a>

&emsp;&emsp;从CUDA 12.0开始，`cudaInitDevice()`和`cudaSetDevice()`调用初始化`runtime`和指定`device`的主要`context`。如果没有这些调用，`runtime`将根据需要隐式使用`device 0`和自初始化来处理其他`runtime`API请求。在计时调用`runtime`函数以及将第一次调用的错误代码解释到`runtime`时，需要记住这一点。在12.0之前，`cudaSetDevice()`不会初始化`runtime`，应用程序通常会使用无操作`runtime`调用`cudaFree(0)`来将`runtime`初始化与其他api活动隔离开来(都是为了计时和错误处理)。

&emsp;&emsp;`runtime`为系统中的每个`device`创建一个CUDA`context`(有关CUDA`context`的更多详细信息，请参阅[`context`](#Title-17.1))。此`context`是`device`的主`context`，并在第一个`runtime`函数中初始化，该函数需要此`device`上的活动`context`。`context`在应用程序的所有`host`线程之间共享。作为`context`创建的一部分，如果有必要，`device`代码将被即时编译(参见[即时编译](#Title-3.1.1.2))并加载到`device`内存中。这一切都是显而易见的。如果需要，例如，对于`driver`API的互操作性，`device`的主要`context`可以从`driver`API访问，如[`runtime`和`driver`APIs之间的互操作性](#Title-17.4)中所述。

&emsp;&emsp;当`host`线程调用`cudaDeviceReset()`时，这会破坏`host`线程当前操作的`device`的主`context`(即，[`device`选择](#Title-3.2.9.2)中定义的当前`device`)。任何拥有此`device`的`host`线程的下一个`runtime`函数调用将为该`device`创建一个新的主`context`。

> __注意__
> CUDA接口使用全局状态，在`host`程序启动期间初始化，在`host`程序终止期间销毁。CUDA`runtime`和`driver`程序无法检测此状态是否无效，因此在程序启动或main后终止期间使用任何这些接口(隐式或显式)将导致未定义的行为。
> 
> 从CUDA 12.0开始，`cudaSetDevice()`会在更改当前`device`的`host`线程后显示初始化`runtime`。之前的CUDA延迟了新`device` `runtime`初始化，直到`cudaSetDevice()`之后进行第一次`runtime`调用。此更改意味着现在检查`cudaSetDevice()`的初始化错误很重要。
> 
> 参考手册中的错误处理和版本管理部分中的`runtime`函数不会初始化`runtime`。

<span id="Title-3.2.2"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory"> 3.2.2、`device`内存</a>

&emsp;&emsp;正如[异构编程](#Title-2.4)中提到的，CUDA编程模型假设一个由`host`和`device`组成的系统，每个`host`和`device`都有自己独立的内存。`kernel`在设备内存之外运行，因此`runtime`提供了分配、释放和复制`device`内存的函数，以及在`host`内存和`device`内存之间传输数据。

&emsp;&emsp;`device`内存可以分配为线性内存或CUDA arrays。

&emsp;&emsp;CUDA数组是针对`texture`获取优化的不透明内存布局。[texture和surface内存](#Title-3.2.14)中有提到。

&emsp;&emsp;线性内存分配在一个单一的统一地址空间中，这意味着单独分配的实体可以通过指针相互引用，例如，在二叉树或链表中。地址空间的大小取决于`host`系统(CPU)和GPU的计算能力

<span id="Table-3.1"></span>

<table class="table-no-stripes docutils align-default" id="id106">
<caption>
<span class="caption-text">Linear Memory Address Space</span>
</caption>
<colgroup>
<col style="width: 48%">
<col style="width: 18%">
<col style="width: 19%">
<col style="width: 15%">
</colgroup>
<thead>
<tr class="row-odd">
<th class="head"></th>
<th class="head"><p>x86_64 (AMD64)</p></th>
<th class="head"><p>POWER (ppc64le)</p></th>
<th class="head"><p>ARM64</p></th>
</tr>
</thead>
<tbody>
<tr class="row-even">
<td><p>up to compute capability 5.3 (Maxwell)</p></td>
<td><p>40bit</p></td>
<td><p>40bit</p></td>
<td><p>40bit</p></td>
</tr>
<tr class="row-odd">
<td><p>compute capability 6.0 (Pascal) or newer</p></td>
<td><p>up to 47bit</p></td>
<td><p>up to 49bit</p></td>
<td><p>up to 48bit</p></td>
</tr>
</tbody>
</table>

> __注意__
> 在计算能力为5.3(`Maxwell`)和更早版本的`device`上，CUDA `drivcer`创建一个未提交的40位虚拟地址预留，以确保内存分配(指针)符合支持的范围。这个预留显示为保留的虚拟内存，但是在程序实际分配内存之前不占用任何物理内存。

&emsp;&emsp;线性内存通常使用`cudaMalloc()`分配，使用`cudaFree()`释放，`host`内存和`device`内存之间的数据传输通常使用`cudaMemcpy()`完成。在`kernel`的向量加法[代码示例](#code-3.2)中，向量需要从`host`内存复制到`device`内存:

<span id="code-3.2"></span>

```C++
// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Host code
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    ...

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    ...
}
```

&emsp;&emsp;线性内存也可以通过`cudaMallocPitch()`和`cudaMalloc3D()`分配。这些函数被推荐用于2D或3D`array`的分配，因为它确保分配被适当地填充以满足[`device`内存访问](#Title-5.3.2)中描述的对齐要求，从而确保在访问行地址或在2D阵列和`device`内存的其他区域之间执行副本时的最佳性能(使用`cudaMemcpy2D()`和`cudaMemcpy3D()`函数)。[代码示例](#code-3.3)分配一个由float组成的宽 x 高的2D数组，并演示如何在`device`代码中循环数组元素:

<span id="code-3.3"></span>

```C++
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr, size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r)
    {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c)
        {
            float element = row[c];
        }
    }
}
```

&emsp;&emsp;[代码示例](#code-3.4)分配一个float的宽度 x 高度 x 深度的3D数组，并演示如何在`device`代码中循环遍历数组元素:

<span id="code-3.4"></span>

```C++
// Host code
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr, int width, int height, int depth)
{
    char* devPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z)
    {
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y)
        {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++x)
            {
                float element = row[x];
            }
        }
    }
}
```

> __注意__
> 为了避免分配过多的内存从而影响系统范围的性能，根据问题大小向用户请求分配参数。如果分配失败，您可以退回到其他较慢的内存类型(`cudaMallocHost()`，`cudaHostRegister()`等)，或者返回一个错误，告诉用户需要多少内存被拒绝。如果您的应用程序由于某些原因无法请求分配参数，我们建议在支持它的平台上使用`cudaMallocManaged()`。

&emsp;&emsp;参考手册列出了所有用于复制线性内存之间的内存的各种函数，用 `cudaMallocPitch()`分配的线性内存，用 `cudaMallocPitch()`或`cudaMalloc3D()`分配的线性内存，CUDA `arrays`以及在`global`或`constant`内存空间声明的变量分配的内存。

&emsp;&emsp;代码示例说明了通过运行时 API 访问全局变量的各种方法:

<span id="code-3.5"></span>

```C++
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
```

&emsp;&emsp;`CudaGetSymbolAddress()`用于检索指向为`global`内存空间中声明的变量分配的内存的地址。`cudaGetSymbolSize()`可获得所分配内存的大小。

<span id="Title-3.2.3"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-l2-access-management"> 3.2.3、`device`内存L2访问管理</a>

&emsp;&emsp;当CUDA`kernel`重复访问`global`内存中的一个数据区域时，这种数据访问可以被认为是持久的。另一方面，如果数据只被访问一次，那么这种数据访问可以被认为是流式的。

&emsp;&emsp;从CUDA11.0开始，具有8.0及以上计算能力的`device`能够影响L2缓存中数据的持久性，有可能提供更高的带宽和对`global`内存的更低延迟访问。

<span id="Title-3.2.3.1"></span>

#### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#l2-cache-set-aside-for-persisting-accesses"> 3.2.3.1、L2缓存用于持久访问的设置</a>

&emsp;&emsp;L2缓存的一部分可以预留出来，用于对`global`内存进行持久化数据访问。持久化访问优先使用这个L2缓存的预留部分，而对`global`内存的正常或流式访问只能在持久化访问未使用时使用L2的这一部分。

&emsp;&emsp;用于持久访问的 L2缓存预留大小可以在限制范围内进行调整:

<span id="code-3.6"></span>

```C++
cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3/4 of L2 cache for persisting accesses or the max allowed*/
```

&emsp;&emsp;当GPU配置为多实例GPU(MIG)模式时，将禁用L2缓存设置功能。

&emsp;&emsp;使用多进程服务(Multi-Process Service，MPS)时，`CudaDeviceSetlimit()`不能更改L2缓存的预留大小。相反，预留大小只能在启动MPS服务器时通过环境变量`CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`指定。

<span id="Title-3.2.3.2"></span>

#### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#l2-policy-for-persisting-accesses"> 3.2.3.2、L2持久访问策略</a>

&emsp;&emsp;访问策略窗口指定一个连续的`global`内存区域和L2缓存中用于该区域内访问的持久性属性。

&emsp;&emsp;[代码示例](#code-3.7)演示如何使用CUDA流设置L2持久访问窗口。

__CUDA Stream Example__

<span id="code-3.7"></span>

```C++
cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                              // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA stream of type cudaStream_t
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

&emsp;&emsp;当`kernel`随后在CUDA流中执行时，访问`global`内存范围`[ptr..ptr+num_bytes)`比访问其他`global`内存位置更有可能持久存储在L2缓存中。

&emsp;&emsp;L2持久性也可以为CUDA`gragh kernel node`设置，如[代码示例](#code-3.8)所示

__CUDA GraphKernelNode Example__

<span id="code-3.8"></span>

```C++
cudaKernelNodeAttrValue node_attribute;                                     // Kernel level attributes data structure
node_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
node_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                            // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
node_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
node_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
node_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA Graph Kernel node of type cudaGraphNode_t
cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);
```

&emsp;&emsp;`hitRatio`参数可用于指定接收`hitProp`属性的访问的分数。在上面的两个示例中，60%的内存访问在`global`内存区域`[ptr..ptr+num_bytes)`具有持久化属性，40%的内存访问具有流属性。哪些特定的内存访问被归类为持久化(`hitProp`)是随机的，其概率接近`hitRatio`，其概率分布取决于硬件架构和内存范围。

&emsp;&emsp;例如，如果L2预留缓存大小为16KB，而`accPolicyWindow`中的`num_bytes`为32KB:
* 当`hitRatio`为0.5时，硬件将随机选择32KB窗口中的16KB作为持久化并缓存在预留的L2缓存区域中。
* 当`hitRatio`为1.0时，硬件将尝试在预留的L2缓存区缓存整个32KB窗口。由于预留区域小于窗口，因此将删除缓存线路，以便将最近使用的32KB数据中的16KB保留在L2缓存的预留部分中。

&emsp;&emsp;因此，可以使用`hitRatio`来避免缓存线路的颠簸，并总体上减少进出L2缓存的数据量。

&emsp;&emsp;`hitRatio`值低于1.0可以用来手动控制数据量不同的`accessPolicyWindow`从并发CUDA流可以缓存在 L2。例如，设置L2的预留缓存大小为16KB; 两个不同CUDA流中的两个并发`kernel`，每个都有16KB的`accessPolicyWindow`，并且都有`hitRatio`值1.0，可能会在竞争共享L2资源时排除对方的缓存线。

<span id="Title-3.2.3.3"></span>

#### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#l2-access-properties"> 3.2.3.3、L2访问属性</a>

&emsp;&emsp;为不同的`global`内存数据访问定义了三种类型的访问属性:


<span id="Title-3.2.3.4"></span>

#### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#l2-persistence-example"> 3.2.3.4、L2持久性示例</a>

<span id="Title-3.2.3.5"></span>

#### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#reset-l2-access-to-normal"> 3.2.3.5、重置L2访问正常</a>

<span id="Title-3.2.3.6"></span>

#### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#manage-utilization-of-l2-set-aside-cache"> 3.2.3.6、管理L2预置缓存利用率</a>

<span id="Title-3.2.3.7"></span>

#### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#query-l2-cache-properties"> 3.2.3.7、查询L2缓存属性</a>

<span id="Title-3.2.3.8"></span>

#### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#control-l2-cache-set-aside-size-for-persisting-memory-access"> 3.2.3.8、控制L2预置缓存大小来持久访问内存</a>

<span id="Title-3.2.4"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory"> 3.2.4、共享内存</a>

<span id="Title-3.2.5"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#distributed-shared-memory"> 3.2.5、分布式共享内存</a>

<span id="Title-3.2.6"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory"> 3.2.6、页面锁定`host`内存</a>

<span id="Title-3.2.8"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution"> 3.2.8、异步并发执行</a>

<span id="Title-3.2.9"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-system"> 3.2.9、多`device`系统</a>

<span id="Title-3.2.9.2"></span>

#### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-selection"> 3.2.9.2、`device`选择</a>

<span id="Title-3.2.12"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#error-checking"> 3.2.12、错误检查</a>

<span id="Title-3.2.13"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#call-stack"> 3.2.13、调用堆栈</a>

<span id="Title-3.2.14"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory"> 3.2.14、`texture`和`surface`内存</a>

<span id="Title-3.2.15"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#graphics-interoperability"> 3.2.15、图形互操作性</a>

<span id="Title-4"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation"> 4、硬件实现</a>

<span id="Title-4.1"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture"> 4.1、`SIMT`架构</a>

<span id="Title-5"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#performance-guidelines"> 5、性能指南</a>

<span id="Title-5.3.2"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses"> 5.3.2、`device`内存访问</a>

<span id="Title-6"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus"> 6、支持CUDA的GPU</a>

<span id="Title-7"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions"> 7、C++语言扩展</a>

<span id="Title-7.26"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#aw-barrier"> 7.26、异步屏障</a>

<span id="Title-7.27"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies"> 7.27、异步数据副本</a>

<span id="Title-7.35"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration"> 7.35、执行配置</a>

<span id="Title-8"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups"> 8、协作组</a>

<span id="Title-9"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism"> 9、CUDA动态并行</a>

<span id="Title-10"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-memory-management"> 10、虚拟内存管理</a>

<span id="Title-11"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator"> 11、流有序内存分配器</a>

<span id="Title-12"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#graph-memory-nodes"> 12、图像存储节点</a>

<span id="Title-13"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix"> 13、数学方法</a>

<span id="Title-14"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-support"> 14、C++语言支持</a>

<span id="Title-15"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching"> 15、纹理方法</a>

<span id="Title-16"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities"> 16、计算能力</a>

<span id="Title-16.6.2"></span>

### <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#independent-thread-scheduling"> 16.6.2、独立线程调度</a>

<span id="Title-16.8"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-9-0"> 16.8、计算能力 9.0</a>

<span id="Title-17"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#driver-api"> 17、Driver API</a>

<span id="Title-17.1"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#context"> 17.1、`context`</a>

<span id="Title-17.4"></span>

## <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#interoperability-between-runtime-and-driver-apis"> 17.4、`runtime`和`Driver` API 之间的互操作性</a>

<span id="Title-18"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-environment-variables"> 18、CUDA环境变量</a>

<span id="Title-19"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming"> 19、统一内存编程</a>

<span id="Title-20"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading"> 20、延迟加载</a>

<span id="Title-21"></span>

# <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#notices"> 21、注意</a>

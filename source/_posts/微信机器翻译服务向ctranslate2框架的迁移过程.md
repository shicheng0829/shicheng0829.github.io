---
title: 微信机器翻译服务向ctranslate2框架的迁移过程
date: 2020-04-30 17:27:33
tags:
---
# 微信机器翻译服务向Ctranslate2框架的迁移过程

## 简介

Ctranslate2是维护OpenNMT模型的公司新出的一种专门针对OpenNMT-py和OpenNMT-tf模型的优化推理引擎，同时支持CPU和GPU。这里的优化主要指两个方面，一是模型的压缩，二是推理的加速。

Ctranslate2具有以下关键特性：

- 高效运行
- 交互式解码
- 模型量化
- 并行翻译
- 动态内存使用
- 自动化的指令集调度
- 轻量的磁盘占用
- 易于使用的翻译API

我们之前机器翻译服务是使用的OpenNMT模型，因此可以很简单的迁移到Ctranslate2上。Ctranslate2的使用方式也很简单，主要分为两步：

1. 转模型，将OpenNMTPy或者OpenNMTf模型转换为该框架支持模型的二进制形式，量化参数支持int8和int16，也可不做量化操作。

   以下是将一个OpenNMTPy模型转换成Ctranslate2 int8量化模型的示例：

   ```python
   import ctranslate2
   converter = ctranslate2.converters.OpenNMTPyConverter(cfg.model)
   output_dir = converter.convert(output_dir="./vx_model",
                                  model_spec=transformer_spec,
                                  quantization=int8,
                                  force=True)
   ```

2. 使用转换后的模型进行翻译。

  ```python
  import ctranslate2
  translator = ctranslate2.Translator("vx_model/")
  translator.translate_batch([["你好", "世界", "!"]])
  ```

## 分析

CTranslate2的核心实现与框架无关。特定于框架的逻辑移至转换步骤，该步骤将训练好的模型序列化为简单的二进制格式。它是以这种方式来兼容不同机器学习框架的model。

其次我觉得需要回答这样一个问题，Ctranslate2的翻译效果为什么快？我觉得关键之处在于"定制"，普遍意义上的神经网络框架比如Pytorch和TensorFlow是针对所有任务的，但是Ctranslate2只是针对机器翻译服务的，可以做一些针对性的优化。

CTranslate2的整体架构可以分为这么几个部分：

1. 通用层

   - 模型格式：模型格式定义了每个model中variable的表示

   - 模型序列化：转二进制格式

2. C++ engine

   - 存储：Ctranslate2使用行优先的存储方式，定义在StorageView Class
   - 抽象层
     - *primitives*：底层计算函数
     - *ops*
     - *layers*
     - *models*
     - *translators*：使用model实现文本翻译逻辑的高阶类
     - *translators pool*：并行计算的translator池，共享同一个model
   - Ops

主要影响Ctranslate2运行速度的有两个参数，分别是`intra_threads`和`inter_threads`：

* intra_threads是每次转换使用的线程数：增加此值可减少延迟。
* inter_threads是并行执行的最大翻译引擎的数量：增加此值以增加吞吐量（由于线程内部的某些内部缓冲区被复制，这也会增加内存使用量）。

我们可以从源码角度看一下这两个参数是在哪个地方起作用的：

```c++
TranslatorPool(size_t num_replicas, size_t num_threads_per_replica, Args&&... args) {
  set_num_threads(num_threads_per_replica);
  _translator_pool.emplace_back(std::forward<Args>(args)...);
  // On GPU, we currently don't benefit much from running instances in parallel, even
  // when using separate streams. This could be revisited/improved in the future.
  if (_translator_pool.back().device() == Device::CUDA)
    num_replicas = 1;
  for (size_t i = 1; i < num_replicas; ++i)
    _translator_pool.emplace_back(_translator_pool.front());
  for (auto& translator : _translator_pool)
    _workers.emplace_back(&TranslatorPool::work_loop,
                          this,
                          std::ref(translator),
                          num_threads_per_replica);
}
```
num_replicas即inter_threads，num_threads_per_replica即intra_threads。

所以inter_threads其实决定了TranslatorPool的大小，即翻译引擎的个数。


```c++
void set_num_threads(size_t num_threads) {
#ifdef _OPENMP
  if (num_threads != 0)
    omp_set_num_threads(num_threads);
#endif
}
```

而intra_threads决定了同一个翻译任务起多少个线程去翻译。

因此在实际部署中，我们采用了inter_threads数为1，intra_threads数等于核数的方案。

## 实现

机器翻译服务现在可以分为这么几个阶段：

- 中英翻译流程：分词，bpe，翻译，delbpe，detruecase，detokenize
- 英中翻译流程：normalize, tokenize, subEntity，转小写，bpe，翻译，delbpe，去空格

替换过程只需要将翻译阶段中的predictor替换成ctranslator2的实现即可，但要注意Ctranslate2框架下的输入和之前的机器翻译服务有些许不同，需要改动一下bpe阶段的输出。

```python
def load_predictor(config_file):
    # model config
    cfg = token_process_tools.TokenProcessor(config_file)
    if with_onmt_py:
        quantize_dynamic = "with quantize_dynamic" if with_quantize_dynamic else "not with quantize_dynamic"
        print("With ONMT ", quantize_dynamic)
        translator = base_model.get_onmt_translator(cfg, with_quantize_dynamic)
        return base_model.BatchPredictor(cfg, translator, debug)
    else:
        print("With ctranslate2 ", ctranslate2_quantization)
        translator = base_model.get_ctranslate2_translator(
            cfg,
            ctranslate2_quantization,
            inter_threads=inter_threads,
            intra_threads=intra_threads)
        return base_model.BatchPredictorWithCtranslate2(cfg, translator, debug)


predictcn = load_predictor('./model/cn2en_config.yml')
predicten = load_predictor('./model/en2cn_config.yml')
```

## 上线

目前还未正式上线，在TKE集群上部署进行小流量测试。因为量化模型会对翻译效果造成一定的影响，因此针对Ctranslate2框架训练的model非常重要。未来会在测试充分的情况下，使用auto-serve正式上线。

## 性能

我们对CTranslate2机器翻译框架替换前后的机器翻译成本做了对比：

测试的机器是2核的机器，每个model测试10个句子，token数目统一是3834。

|                            | 时间   | QPS    | 处理1M token的时间(h) | 处理1M token的成本(元) |
| -------------------------- | ------ | ------ | --------------------- | ---------------------- |
| 长句子(V2[onmt，无量化])   | 75.33  | 0.1327 | 5.4579                | 0.5542                 |
| 长句子(V2[onmt，有量化])   | 38.46  | 0.2600 | 2.7865                | 0.2829                 |
| 长句子(V3[onmt，无量化])   | 181.69 | 0.0550 | 13.1640               | 1.3367                 |
| 长句子(V3[onmt，有量化])   | 95.07  | 0.1051 | 6.8879                | 0.6994                 |
| 长句子(V4[onmt，无量化])   | 149.83 | 0.0667 | 10.8557               | 1.1023                 |
| 长句子(V4[onmt，有量化])   | 77.40  | 0.1291 | 5.6082                | 0.5695                 |
| 长句子(V4[ctrans2，fp32])  | 139.53 | 0.0716 | 10.1094               | 1.0265                 |
| 长句子(V4[ctrans2，int16]) | 102.13 | 0.0979 | 7.4001                | 0.7514                 |
| 长句子(V4[ctrans2，int8])  | 70.26  | 0.1423 | 5.0910                | **0.5169**             |

可以看出替换成Ctranslate2的机器翻译框架，在int8的量化情况下要比ONMT的量化模型节省了9.6%的成本。

## 总结和展望

我们基于Ctranslate2这种全新的机器翻译框架实现了机器翻译服务上的predict模块，翻译的性能有了一些提升，可以节省下一些机器的成本。未来会对model进行针对Ctranslate2框架进一步的调优，并且会进行更加充分的测试。

关于进一步的调优工作，如果是在GPU上进行的机器翻译服务，可以对CUDA caching allocator的一些参数进行针对性的调优，如bin_growth，min_bin，max_bin和max_cached_bytes等参数。

这个工作在josephyu和florianzhao指导下进行，感谢他们。
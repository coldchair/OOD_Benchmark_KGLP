# OOD_Benchmark_KGLP

### 如何使用：

```
git clone https://github.com/coldchair/OOD_Benchmark_KGLP.git
```

进到对应的目录，从清华云下载：models 和 image（可以不下） 文件夹。

### 仓库结构介绍：

下面为一个示例仓库：

|-- root
    |-- .gitignore
    |-- README.md
    |-- bash
    |-- image
    |   |-- transE_WN18
    |       |-- s_plus_o_100_buckets.svg
    |-- models
        |-- transE_WN18
            |-- ranks.csv
            |-- test.txt
            |-- train.txt
            |-- valid.txt
            |-- models.pkl
            |-- get_ranks.py * # 用于获得performance.txt 和 ranks.csv
            |-- main.py * # 用于训练
            |-- performance.txt * # test 的各个指标
            |-- logs * # tensorboard 存储目录


####  models 目录

- 所有模型需要被放到该目录下。
- 一个训练好的模型存放在 `models/{modelsName_datasetName}/` 目录下
- 上表中不带 * 为必须存放的东西，带 * 为选择存放（仅供参考），你也可以存其它的东西。
- **<u>save 的模型必须命名为 `models.pkl`，里面的那个也要叫 `models.pkl_metadata.ampkl`。</u>**
- **<u>`ranks.csv` 为 test 集中 head 和 tail 的预测排名，是一个 $n \times 2$  的逗号分隔的 csv 表。</u>**
- 如果你已经 save 好了，可以复制 `transE_WN18/get_ranks.py` 到你的目录下，运行它会生成  `ranks.csv` 和 `performance.txt`。

#### bash 目录

**<u>已实现工具和类（如果有更新请写在下表，接口自己读注释）：</u>**

| 名称                    | 作用                       |
| ----------------------- | -------------------------- |
| dataset.py 下 Dataset类 | 读入、存图                 |
| utils/dir               | dir 工具                   |
| utils/evalution         | mr,mmr,hit@n               |
| utils/read_ranks        | 读入ranks.csv 转为 ndarray |
| degree/get_degree       | 返回三个度数 list          |

- 在 bash 目录下运行 `test.py` 即可生成按头尾度数和排序，划分100个 buckets，纵坐标为 MMR 的图。
- 你可以参考 `test.py` 和 `degree/s_plus_o_100_buckets.py` 利用工具的方法来写你的脚本，在写脚本的过程注意最好不要修改已有的文件，**<u>一定保证最初的 `test.py` 能正常运行</u>**，有没有好心人写点测试o(╥﹏╥)o
- `bash/` 下的运行主脚本命名建议带有名字缩写前缀，不然可能会撞车。

#### image 目录

图片需要被存在 `image/{modelsName_datasetName}/` 目录下。

注意命名，不要和别人的重复了。

### 其它：

`.gitignore` 已经添加了 `models/` 和 `image/` 和一些常规的 ignore，请注意你想上传的文件（夹）不要出现在 .`gitignore` 里。

注意大文件不要传到 Github 上来。

**<u>自己创建新的分支写， 每周开会时 merge 分支。</u>**
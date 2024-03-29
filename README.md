# OOD_Benchmark_KGLP

### 9.20更新

MEIM-KGE 的 ranks 文件已更新。

Anyburl 和 TuckER 添加。



简单讲一下：MEIM-KGE 的 rank 是怎么得到的：

假设预测 h, r, t 中的 t

raw：对所有 train 和 test 中出现过的 entity，都得到评分，求 t 在其中的排名。

filter：依旧是对所有 entity 求评分，但是 只考虑在 $\{x | x\in(h,r,x) \in Train ∪ Test\}$ 中的 t 的排名。

不难发现，filter 的结果，受 test 中其它边影响小，因为大部分边在 train 中，而 train不变。

反向边同理。

排名策略（min)：同分时取排名最小

排名策略（max）：同分时取排名最大

排名策略（mean)：求 min 和 max 的平均值。

MEIM 采用 mean 策略。



### 9.10 更新

模型请从清华云上扒下来：https://cloud.tsinghua.edu.cn/group/47256/

MEIM 的环境见 https://github.com/tranhungnghiep/MEIM-KGE，也可以到 realcenter KG-new 上的 shared_disk 上找。

目前脚本不支持 MEIM 直接运行（主要没环境），想单独运行，可以直接运行 sh bash.sh 。


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
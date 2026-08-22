# 瑜伽体式教练 / Yoga Coach

用摄像头看你的瑜伽练习，实时打分并给出**具体的**纠正建议——不是"做得不错"，而是"前膝移回脚踝正上方，别超过脚尖"。

```
战士二式 · 右侧
Virabhadrasana II

65 分  ████████████░░░░░░░░

调整建议
1. 前膝移回脚踝正上方，别超过脚尖      前膝对准脚踝: 0.60× → ≤0.22×
2. 后腿蹬直，后脚外缘压实地面          后腿伸直: 142° → ≥165°
3. 前膝屈得太深了，小腿回到垂直        前膝屈度: 65° → 80~110°
```

---

## 快速开始

```bash
git clone <this repo> && cd Faye_Pivlot
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m yoga_coach          # 打开默认摄像头，自动识别体式
```

第一次运行会自动下载约 6 MB 的姿态识别模型，之后就不再需要联网。

**Linux 用户**还需要装几个系统库（macOS 和 Windows 不用）：

```bash
sudo apt install libegl1 libgles2 libgl1 libglib2.0-0
```

窗口里的按键：`q` 退出 · `d` 显示/隐藏每项检查的具体数值 · `r` 重新计时。

## 怎么用

```bash
python -m yoga_coach --pose warrior2        # 只练战士二式，不自动切换
python -m yoga_coach --list-poses           # 看看支持哪些体式、摄像头该放哪
python -m yoga_coach --source practice.mp4  # 回看一段自己的练习录像
python -m yoga_coach --source photo.jpg     # 分析一张照片，打印完整报告
python -m yoga_coach --record out.mp4       # 把带标注的画面录下来
python -m yoga_coach --speak                # 语音播报建议（需要 pip install pyttsx3）
python -m yoga_coach --model heavy          # 换更准但更慢的模型
python -m yoga_coach --lang en              # 英文界面
```

`--speak` 挺有用：下犬式的时候你根本看不到屏幕。

### 摄像头怎么放

| 体式 | 摄像头位置 |
| --- | --- |
| 山式、树式 | 正前方，能拍到从头到脚 |
| 战士一/二、三角式 | 正前方，你侧身站 |
| 下犬式、平板支撑、椅子式 | 身体侧面 |

关键是**全身入镜**。身体只拍到一半时，程序会直接告诉你"身体没有完整入镜"，而不是拿仅剩的几个还量得到的角度硬凑一个分数出来。

### 支持的体式

| key | 体式 | Sanskrit |
| --- | --- | --- |
| `mountain` | 山式 | Tadasana |
| `tree` | 树式 | Vrksasana |
| `warrior1` | 战士一式 | Virabhadrasana I |
| `warrior2` | 战士二式 | Virabhadrasana II |
| `triangle` | 三角式 | Trikonasana |
| `downdog` | 下犬式 | Adho Mukha Svanasana |
| `plank` | 平板支撑 | Phalakasana |
| `chair` | 椅子式 | Utkatasana |

左右不用告诉程序：战士二式左腿在前还是右腿在前，它自己判断（两种读法都算一遍，取分高的那个）。

---

## 它是怎么判断的

```
摄像头 ──▶ MediaPipe 姿态识别 ──▶ 33 个关键点
                                      │
                                      ▼
                          几何测量（关节角、水平度、
                          相对位置，全部按躯干长度归一化）
                                      │
                                      ▼
                   每条规则一个目标区间 → 分数 + 一句人话建议
                                      │
                                      ▼
                     跨帧平滑、按严重程度排序、只显示最重要的 3 条
```

一个体式就是一组规则。战士二式里"前膝屈度"这条长这样：

```python
Check(
    key="front_knee_bend",
    label=Text("前膝屈度", "Front knee bend"),
    metric=m.joint_angle("{s}_hip", "{s}_knee", "{s}_ankle"),
    low=80.0, high=110.0, falloff=35.0, weight=1.4,
    when_low=Text("前膝屈得太深了，小腿回到垂直", "Too deep -- shin back to vertical"),
    when_high=Text("前腿再屈深一点，大腿趋向平行地面", "Bend the front knee deeper"),
    focus=("{s}_knee",),
)
```

* 落在 `80~110°` 之间得满分，超出后线性扣分，偏离 `falloff` 度就归零。
* 偏低和偏高给**不同的**建议——"膝盖角度不对"帮不了任何人。
* `focus` 让画面上对应的关节标红，建议才有指向。
* `{s}` 是"工作侧"（战士系列的前腿、树式的支撑腿），`{o}` 是另一侧。

几个刻意的设计：

* **距离都除以躯干长度**，所以你走近走远，建议不会变。
* **看不见的关键点不算错**，只是跳过，并在报告里列出来——遮挡不该被判成姿势错误。
* **建议至少停留 1.2 秒**才换下一条，不然每帧都在跳，根本没法读；但已经改对的那条会立刻消失。
* **自动识别体式带迟滞**：新体式要连续 8 帧、且分数高出 6 分以上才切换，否则战士一和战士二会一直互相抢。

## 加一个新体式

在 `yoga_coach/poses.py` 里写一个 `PoseSpec` 加进 `POSES` 就行，其他地方都不用动：

```python
BOAT = PoseSpec(
    key="boat",
    name=Text("船式", "Boat"),
    sanskrit="Navasana",
    view=Text("摄像头放在身体侧面", "Camera to your side"),
    symmetric=False,
    checks=(
        Check(
            key="knee_straight",
            label=Text("腿伸直", "Legs straight"),
            metric=m.joint_angle("{s}_hip", "{s}_knee", "{s}_ankle"),
            low=160.0, falloff=35.0,
            when_low=Text("小腿再伸直一些", "Straighten the shins"),
            focus=("{s}_knee",),
        ),
        # ...
    ),
)
```

可用的测量在 `yoga_coach/metrics.py`：关节角、与竖直/水平的夹角、左右倾斜、水平/垂直间距、直线偏移。

## 测试

```bash
pip install pytest
python -m pytest tests -q
```

108 个测试，不需要摄像头也不需要 MediaPipe：测试用的是手写的"火柴人"骨架（`tests/figures.py`），坐标都在注释里标好了。这样"前膝超过脚踝 40 度会触发哪条建议"是可以精确断言的，不用指望某段录像里刚好还有这个错误。

---

## 几句实话

* **这不能替代老师。** 程序看的是二维投影，判断不了脊柱内部的排列、肌肉是否发力、你今天的肩膀能不能到那个位置。它擅长的是抓明显的对位问题——膝盖过脚尖、后腿没蹬直、髋歪了。
* **目标区间是通用课的口令，不是你的个人标准。** 柔韧度、比例、伤病史都会改变什么叫"正确"。分数低不等于做错了。
* **疼就停。** 任何时候都以身体的感觉为准，不要为了把分数拉到 90 而硬压。
* **孕期、伤后康复、有椎间盘或关节问题**，请听医生和线下老师的，别听这个程序的。
* 画面不会离开你的电脑：没有上传，没有联网（除了第一次下载模型），没有任何录制默认开启。

## 项目结构

```
yoga_coach/
  geometry.py    坐标计算：角度、距离、平滑
  landmarks.py   33 个关键点的命名与单帧骨架
  metrics.py     可以测什么
  checks.py      一条规则 = 一个测量 + 一个目标区间 + 一句建议
  poses.py       体式库
  evaluator.py   给一帧打分、猜是哪个体式
  session.py     跨帧：平滑、体式跟踪、保持计时
  detector.py    MediaPipe 封装（唯一依赖 MediaPipe 的文件）
  render.py      画面叠加（中文文字走 Pillow，OpenCV 画不了中文）
  console.py     终端输出
  voice.py       可选语音
  cli.py         命令行
tests/
  figures.py     手写骨架
```

# 仓库约定

仓库里有**两个独立的工具**，共用 MediaPipe 底座但代码互不引用：

| 包 | 做什么 | 文档 | 约定写在哪 |
|----|--------|------|-----------|
| `yoga_coach` | 摄像头 → 关键点 → 几何规则 → 实时纠正建议 | `docs/yoga_coach.md` | 本文件以下全部 |
| `yoga_grid` | 练习视频 → 抓正位帧 → 拼九宫格 | `README.md`（仓库首页） | `README.md` 的「内置体式模板」「测试」两节 |

**两边的体式模板是分开的**（`yoga_coach` 9 个、`yoga_grid` 20 个），目标值和容差
各自校准，改一边不会影响另一边。想让某个改动同时生效，两边都要改。

一个已经付出过代价的教训：两套代码在**并行解决同一个问题**（体式几何判别）。
`yoga_grid` 记的是「上犬式被三角伸展式认走」，`yoga_coach` 记的是「上犬式被
平板/山式认走」。动其中一边的判别逻辑时，去另一边看看同样的坑有没有记录。

`tests/` 是两个工具共用的一个目录，**文件名前缀区分归属**：`test_grid_*` 属于
`yoga_grid`，其余属于 `yoga_coach`。两边的测试都不需要 MediaPipe。

---

以下是 `yoga_coach` 的约定。

## 改完代码必须做的事

**1. 同步教练文档。** `docs/yoga_coach.md` 里体式表、每个体式的检查项和目标区间、
模块清单、规则条数都是从代码生成的，不要手改：

```bash
python tools/update_readme.py
```

（这个脚本只管 `docs/yoga_coach.md`。仓库首页 `README.md` 是 `yoga_grid` 的，
全手写，没有生成区块。）

忘了跑会被 `tests/test_readme.py` 拦下来。生成区块之外的散文（技术方案、
训练指导的文字部分）是手写的——改了功能、参数、运行方式，要一并手动更新，
特别是这几处：

- 「功能列表」和「命令行参数速查」两张表 —— 增删 CLI 参数或运行模式时
- 「训练指导 → 怎么读屏幕上的反馈」 —— 改了阈值、颜色分档、按键时
- 「技术方案 → 时序处理」 —— 改了平滑系数、驻留时间、切换迟滞时

**2. 跑测试和静态检查。**

```bash
pip install -r requirements-dev.txt     # 不含 MediaPipe，跑测试用不上
python -m pytest tests -q               # 两个包的测试都在这里
python -m pyflakes yoga_coach yoga_grid tests tools
```

**3. 动了 `detector.py` 或依赖版本，额外跑一次冒烟测试。** 单元测试碰不到
MediaPipe，API 改名、wheel 装不上、模型 URL 变了都不会让测试失败：

```bash
pip install -r requirements.txt
python tools/smoke_test.py
```

CI（`.github/workflows/ci.yml`）两个 job 跑的就是上面这些：`test` 是轻依赖的
pyflakes + pytest + 教练文档同步检查（3.10/3.11/3.12），`smoke` 装完整依赖跑
`tools/smoke_test.py`，并加载两个包的 CLI——`yoga_grid` 里 import cv2 的那几个
模块（`cli` / `extract` / `grid` / `compare` / `faces` / `compat`）轻依赖 job
看不到，只有这一步能发现它们导入坏了。改了依赖或 Python 版本支持范围，
记得同步 workflow。

## 分层：依赖方向严格单向

```
geometry → landmarks → metrics → checks → poses → evaluator → session
                                                        ↑
                          detector / render / console / voice / cli
```

- **只有 `detector.py` 能 import mediapipe，只有 `render.py` / `cli.py` 能 import cv2。**
  这不是风格偏好：正因为规则层不碰摄像头栈，测试才能跑在手写骨架上、
  0.4 秒跑完、不需要摄像头。加依赖前先想清楚放在哪一层。
- 新的测量方式加到 `metrics.py`，新体式加到 `poses.py`，两边都不该出现 I/O。

## 写体式规则时

- 目标区间是通用课的口令，宁可宽一点。区间太严会让人为了刷分做出危险动作。
- 低于区间和高于区间要给**不同**的提示语。"角度不对"帮不了任何人。
- 注意重力方向：站姿在几何上就是一个完美的平板支撑（肩髋踝共线、四肢伸直）。
  新体式和已有体式几何相似时，要补一条判别性的检查，否则自动识别会认错。
- 距离类测量一律用 `metrics` 里按躯干长度归一化的函数，不要直接用像素或归一化坐标差。
- 改了区间数值，`tests/test_poses.py` 里的合成骨架断言会立刻告诉你改过头了没有。

## 测试

`tests/figures.py` 是手写坐标的火柴人骨架，每个姿势的几何推导写在注释里。
新体式请配一个理想骨架 + 至少一个"典型错误"骨架，断言那个错误会触发**指定的**
建议 key，而不只是断言分数变低。

## 语音

同样按"策略 / IO"拆开：**说什么**在 `announce.py`（纯逻辑，用合成骨架测），
**怎么发声**在 `voice.py`（pyttsx3、挑音色、后台线程）。改播报规则只动前者。

四条容易踩的：

- 只念错误是不够的。做对时也要出声，否则听者分不清"做对了"和"没认出我"。
- 播报分强制和非强制：体式切换、到位、完成一组必须送达（`force=True`）；
  纠正建议在引擎忙时丢弃，但要调 `announcer.undo()` 让它下一帧重试，
  否则会被节流窗口静音掉。
- **Windows 上每句话必须新建引擎**（`voice.py` 的 `default_engine_mode()`）。
  pyttsx3 + SAPI5 只发得出第一句，之后 `runAndWait()` 照常返回、不抛异常、
  终端全绿，但没有声音——异常处理抓不到。已在真机确认：fresh 三句全响，
  persistent 只响第一句。别为了"简化"改回共用一个引擎。
- **引擎必须由播报线程自己创建。** SAPI5 是 COM 对象，归创建它的线程所有；
  在主线程建好、交给后台线程用，Windows 上播完一句就开始抛异常。播报循环
  也绝不能 `return`——那样队列没人读，之后每一句都静默丢弃。
- 改完用 `--speak-test` 验证（两种模式各连播三句）。**它不会说"语音正常"**——
  程序没有麦克风，只能确认调用没报错。别让任何代码去断言它观测不到的事。

## 中文与文案

界面文案用 `Text(zh, en)` 双语容器，两版都要写。
中文渲染走 Pillow（`cv2.putText` 画不了中文），找不到 CJK 字体时整体回退英文。

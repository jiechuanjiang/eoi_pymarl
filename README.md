This is the codebase of improved EOI on general SMAC tasks. We have also published EOI on sparse-reward so_many_baneling in this [repo](https://github.com/jiechuanjiang/EOI_on_SMAC).

## Run an experiment
```
python3 src/main.py --config=eoi with explore_ratio=0.2 --env-config=sc2
```

Two parameters explore_ratio and episode_ratio control the strength of EOI exploration. The improved version of EOI could be seen as a multi-agent exploration method.

## Results

<img src="results1.png" alt="EOI" width="800">

|               | 2c_vs_64zg | 3s_vs_5z | 5m_vs_6m |
| ------------- | ---------- | -------- | -------- |
| explore_ratio | 0.2        | 0.8      | 0.2      |

The agents are more likely to benefit from individualized behaviors if the trajectory is longer.

3s_vs_5z and 5m_vs_6m are tested on SC2.4.10, and 2c_vs_64zg are tested on SC2.4.16, since the 2c_vs_64zg on SC2.4.10 is too easy.

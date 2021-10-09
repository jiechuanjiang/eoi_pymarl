This is the codebase of improved EOI on general SMAC tasks. We have also published EOI on sparse-reward so_many_baneling in this [repo](https://github.com/jiechuanjiang/EOI_on_SMAC).

## Run an experiment
```
python3 src/main.py --config=eoi with explore_ratio=0.2 --env-config=sc2
```

Two parameters explore_ratio and episode_ratio control the strength of EOI exploration. The improved version of EOI could be seen as a multi-agent exploration method.

## Results

<img src="results.png" alt="EOI" width="500">

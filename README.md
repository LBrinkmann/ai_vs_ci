# ai_vs_ci

## Helpfull Cluster command

```
scancel $(squeue -o '%j %.18i' -h | grep 'on_off' |  awk '{print $2}')
```

scancel $(squeue -o '%j %.18i' -h | grep 'deep_agent_type_ai' | awk '{print $2}')

# data

The training data used by `intent-classifier` should be in yml format. For example:

```
- intent: confusion
  examples:
    - wait what?
    - huh? im confused
    ...

- intent: neutral
  examples:
    - Alright, let's see what happens
    - I'm not ready
    - We should continue with the next part
    ...
```
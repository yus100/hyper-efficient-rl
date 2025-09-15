# Hyper-Efficient RL

goal: improve math reasoning in small language models using supervised fine-tuning and reinforcement learning

## model
start from a base model with no instruction tuning (llama 3 8b base or qwen3 4b base)

## method
- sft with lora on gsm8k and math subsets  
- ppo with curriculum learning (speed)  
- reward shaping with length penalty  

## eval
benchmarks: gsm8k, math, svamp  
metrics: accuracy, token count, latency  


## refs
- transformers + trl: [huggingface trl library](https://arxiv.org/abs/2306.09683)  
- rl on small models: [openrlhf experiments](https://arxiv.org/abs/2402.01306)  
- length-aware optimization: [rewarding shorter reasoning chains](https://arxiv.org/abs/2402.10896)  
- curriculum learning for rl: [speed curriculum](https://arxiv.org/abs/2407.01082)  
- math reasoning datasets: [gsm8k](https://arxiv.org/abs/2110.14168), [math](https://arxiv.org/abs/2103.03874)

# VLA with InternVM 3.5 Lerobot Fork

## train 

Add DDP training feature.

```bash
torchrun --standalone --nproc_per_node=8 -m lerobot.scripts.train --policy.type=internvla --policy.knowledge_insulation false --policy.train_expert_only true --policy.use_discrete_aux false --dataset.repo_id=k1000dai/libero --batch_size=8  --steps=150000 --policy.repo_id=k1000dai/internvla_libero --output_dir=out_internvla_no_KI  --job_name=libero_internvla_no_KI --wandb.enable=true
```

See src/lerobot/policies/internvla/configuration_internvla.py

for all configuration options.

Try to ad Knowledge Insulation (KI) and it works well but show slightly worse performance than no KI. (maybe requires more tuning ? )


## eval 

```bash
python eval_libero/evaluation_libero.py
``` 

See `eval_libero/evaluation_libero.py` for details.

Libero model file is here : https://huggingface.co/k1000dai/internvla_libero

Dataset is here : https://huggingface.co/datasets/k1000dai/libero


## Results

- Spatial: 86
- Object: 97
- Goal: 85
- LIBERO-10: 72

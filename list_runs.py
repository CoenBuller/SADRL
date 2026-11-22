import wandb

api = wandb.Api()
runs = api.runs("coen-sadrl/sadrl")

for r in runs:
    print(f"id={r.id} | name={r.name} | state={r.state} | env={r.config.get('ENV_NAME')} | seed={r.config.get('SEED')}")
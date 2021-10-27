import wandb
wandb.login()
from datetime import datetime
from pytz import timezone
def wandb_args_init(wandb_args, model_args):     
    wandb_args.tags = list(wandb_args.tags)
    if not wandb_args.group :
        wandb_args.group = model_args.model_name_or_path

    if not wandb_args.tags: 
        wandb_args.tags = [wandb_args.author,model_args.model_name_or_path]
    else:
        wandb_args.tags.append(wandb_args.author)
        wandb_args.tags.append(model_args.model_name_or_path)

    if model_args.model_name_or_path == "./models/train_dataset/" :
        wandb_args.group = "eval"
        wandb_args.tags = ["eval"]

    if not wandb_args.name:
        wandb_args.name = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    wandb_args.name = wandb_args.author +"/"+wandb_args.name
    print(f"Create {wandb_args.name} chart in wandB...")
    print(f"WandB {wandb_args.entity}/{wandb_args.project} Project Group and tages [{wandb_args.group},{wandb_args.tags}]")
    return wandb_args


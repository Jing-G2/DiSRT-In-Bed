from utils import logger
from torch.optim import lr_scheduler


def get_scheduler(optimizer, lr_policy, total_epochs):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        lr_policy          -- the name of learning rate policy:
                                linear | cosine | exponential | multistep | step | plateau | none
        total_epochs       -- the total number of epochs to train the network
    """
    try:
        if total_epochs <= 0:
            raise ValueError(f"total_epochs must be positive, but got {total_epochs}")

        if lr_policy == "linear":

            def lambda_rule(epoch):
                if epoch < total_epochs:
                    lr_l = 1 - epoch / (total_epochs + 1)
                else:
                    lr_l = 1 / (epoch + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif lr_policy == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        elif lr_policy == "exponential":
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        elif lr_policy == "multistep":
            scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[total_epochs // 2, total_epochs * 2 // 3],
                gamma=0.5,
            )
        elif lr_policy == "step":
            scheduler = lr_scheduler.StepLR(
                optimizer, step_size=total_epochs, gamma=0.1
            )
        elif lr_policy == "plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
            )
        elif lr_policy == "none":
            logger.log("No learning rate policy is selected, using the default one.")
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    except Exception as e:
        logger.log(
            f"Use the default policy because learning rate policy {lr_policy} is not recognized: {e}"
        )
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    return scheduler

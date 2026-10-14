import random
import time
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from .metrics import missed_eval_np, calc_quantile_CRPS

def train(
    model,
    config,
    args,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
    current_time=None,
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model_{}.pth".format(current_time)

    p1 = int(0.5 * config["epochs"])
    p2 = int(0.75 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    alpha = config["alpha"]
    beta = config["beta"]
    gamma = config.get("gamma", 1.0)
    eta = config.get("eta", 0.0)

    train_losses = []
    valid_losses = []

    train_start_time = time.time()

    for epoch_no in range(config["epochs"]):
        epoch_start_time = time.time()
        avg_loss, avg_loss_noise, avg_loss_cons, avg_loss_freq_cons, avg_loss_cic = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        model.train()
        for batch_no, train_batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(train_batch)
            if len(outputs) == 4:
                loss_noise, loss_cons, loss_cic, loss_freq_cons = outputs
                loss = (
                    alpha * loss_noise
                    + beta * loss_cons
                    + eta * loss_cic
                    + gamma * loss_freq_cons
                )
            elif len(outputs) == 3:
                loss_noise, loss_cons, loss_freq_cons = outputs
                loss_cic = torch.tensor(0.0)
                loss = alpha * loss_noise + beta * loss_cons + gamma * loss_freq_cons
            else:
                loss_noise, loss_cons = outputs
                loss_cic = torch.tensor(0.0)
                loss_freq_cons = torch.tensor(0.0)
                loss = alpha * loss_noise + beta * loss_cons
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_loss_noise += loss_noise.item()
            avg_loss_cons += loss_cons.item()
            avg_loss_freq_cons += loss_freq_cons.item()
            avg_loss_cic += (
                loss_cic.item() if isinstance(loss_cic, torch.Tensor) else 0.0
            )
        lr_scheduler.step()
        train_loss = avg_loss / (batch_no + 1)
        train_loss_noise = avg_loss_noise / (batch_no + 1)
        loss_cl = avg_loss_cons / (batch_no + 1)
        loss_freq_cl = avg_loss_freq_cons / (batch_no + 1)
        loss_cic_avg = avg_loss_cic / (batch_no + 1)
        train_losses.append(train_loss)

        epoch_elapsed = time.time() - epoch_start_time

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                for batch_no, valid_batch in enumerate(valid_loader):
                    loss = model(valid_batch, is_train=0)
                    avg_loss_valid += loss.item()
                valid_loss = avg_loss_valid / (batch_no + 1)
                valid_losses.append(valid_loss)
            print(
                "Epoch {}: train loss = {:.6f}, valid loss = {:.6f} [Time: {:.1f}s]".format(
                    epoch_no + 1, train_loss, valid_loss, epoch_elapsed
                )
            )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                torch.save(model.state_dict(), output_path)
        else:
            print(
                "Epoch {}: train loss = {:.6f}, cic = {:.6f} [Time: {:.1f}s]".format(
                    epoch_no + 1, train_loss, loss_cic_avg, epoch_elapsed
                )
            )

    total_train_time = time.time() - train_start_time
    print("\nTotal training time: {:.2f}s".format(total_train_time))

    return train_losses, valid_losses, total_train_time

def evaluate(
    model,
    test_loader,
    nsample,
    scaler,
    mean_scaler,
    save_result_path,
    current_time=None,
):
    with torch.no_grad():
        model.eval()
        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        imputed_data = []
        groundtruth = []
        eval_mask = []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)
                c_target = c_target.permute(0, 2, 1)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)

                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                output = samples_median.values * scaler + mean_scaler
                X_Tilde = c_target * scaler + mean_scaler
                eval_M = eval_points
                imputed_data.append(output.cpu().numpy())
                groundtruth.append(X_Tilde.cpu().numpy())
                eval_mask.append(eval_M.cpu().numpy())

        results = {}
        results["imputed_data"] = np.concatenate(imputed_data, axis=0)
        results["groundtruth"] = np.concatenate(groundtruth, axis=0)
        results["eval_mask"] = np.concatenate(eval_mask, axis=0)

        mae, rmse, mape, mse, r2 = missed_eval_np(
            results["imputed_data"],
            results["groundtruth"],
            1 - results["eval_mask"],
        )

        all_target = torch.cat(all_target, dim=0)
        all_evalpoint = torch.cat(all_evalpoint, dim=0)
        all_observed_point = torch.cat(all_observed_point, dim=0)
        all_observed_time = torch.cat(all_observed_time, dim=0)
        all_generated_samples = torch.cat(all_generated_samples, dim=0)

        CRPS = calc_quantile_CRPS(
            all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
        )

        print(
            "mae = {:.3f}, rmse = {:.3f}, mape = {:.3f}%, mse = {:.3f}, r2 = {:.3f}, CRPS = {:.4f}".format(
                mae, rmse, mape * 100, mse, r2, CRPS
            )
        )

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape * 100,
        "mse": mse,
        "r2": r2,
        "CRPS": CRPS,
    }

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

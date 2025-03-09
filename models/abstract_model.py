from typing import Dict, List
from pathlib import Path
import os

from tqdm import tqdm, trange
import cv2
import numpy as np
from oml.retrieval import RetrievalResults, AdaptiveThresholding
from oml.metrics import calc_retrieval_metrics_rr
from oml.inference import inference
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import normalize


def save_model_dict(model, path: str, epoch):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), path + "/" + str(epoch) + ".pth")


class ABCModel(nn.Module):
    def training_oml(
        self,
        n_epochs: int,
        save_frequency: int,
        device: str,
        dataset,
        model,
        criterion,
        optimizer,
        path2weights,
        scheduler,
        loss_device: str = "cpu",
    ):

        for epoch in range(n_epochs):
            pbar = tqdm(DataLoader(dataset.train, batch_sampler=dataset.sampler))
            pbar.set_description(f"epoch: {epoch}/{n_epochs}")
            model.model.train()
            for batch in pbar:
                embeddings = model(batch["input_tensors"].to(device)).to(loss_device)
                embeddings = normalize(embeddings)
                loss = criterion(embeddings, batch["labels"].to(loss_device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix(criterion.last_logs)
            if scheduler:
                scheduler.step()
            if (epoch + 1) % save_frequency == 0:
                self.validation_oml(model, dataset)
                save_model_dict(model, path2weights, epoch)

    def validation_oml(self, model, dataset):

        embeddings = inference(
            model, dataset.val, batch_size=128, num_workers=0, verbose=True
        )
        rr = RetrievalResults.from_embeddings(embeddings, dataset.val, n_items=10)
        rr = AdaptiveThresholding(n_std=2).process(rr)
        rr.visualize(query_ids=[2, 1], dataset=dataset.val, show=True)
        results = calc_retrieval_metrics_rr(rr, map_top_k=(10,), cmc_top_k=(1, 5, 10))

        for metric_name in results.keys():
            for k, v in results[metric_name].items():
                print(f"{metric_name}@{k}: {v.item()}")

    def train_loop(
        self,
        model_dict: dict,
        losses: Dict[str, int],
        dataloaders: dict,
        metrics_logger,
        n_epochs: int,
        path2weights: str,
        save_frequency: int,
        device: str,
        start_epoch: int = 0,
        scheduler_epoch: bool = True,
        scheduler_freq: int = 1,
    ):
        """
        model_dict = {
            "model": torch.nn.Module,
            "optimizer": torch.optim.Adam,
            "scheduler": torch.optim.lr_scheduler,
            "epoch": int
        }
        dataloader: torch.utils.data.Dataloader
        """
        model = model_dict["model"]
        optimizer = model_dict["optimizer"]
        scheduler = model_dict["scheduler"]

        loss_func = TotalLoss(losses, device=device)
        model = model.to(device)

        best_nse = 1e10
        best_epoch = 0
        try:
            for epoch in trange(start_epoch, n_epochs, 1):
                model.train()
                for _ in tqdm(dataloaders["train"]):

                    optimizer.zero_grad()
                    loss = loss_func()

                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        metrics_logger.compute_metrics()
                    if (not scheduler_epoch) and scheduler:
                        scheduler.step()

                if scheduler and scheduler_epoch and (epoch % scheduler_freq == 0):
                    scheduler.step()
                metrics_logger.log_metrics(mode="train", epoch=epoch)

                model.eval()
                self.val_loop(model, dataloaders["val"], metrics_logger, device)
                curr_nse = np.mean(metrics_logger.get_metrics()["NSE"])
                if (curr_nse < best_nse) and ((epoch + 1) % save_frequency == 0):
                    best_epoch = epoch + 1
                metrics_logger.log_metrics(mode="val", epoch=epoch)
                if (epoch + 1) % save_frequency == 0:
                    save_model_dict(
                        model, optimizer, scheduler, epoch + 1, path2weights
                    )
        except KeyboardInterrupt:
            print("Keyboard Interrupt: Saving best model")

        return best_epoch

    def val_loop(
        self,
        model,
        dataloader,
        metrics_logger,
        device,
        interpolate: bool = False,
        inference: bool = False,
        path2save: Path | str | None = None,
        imgs_names: List[str] | None = None,
        hdr: bool = False,
        show: bool = False,
    ):
        i = 0
        """
        path2save: str, path to save reconstructed images
        """
        batch_size = 4
        for gt_hr_hsi, gt_lr_hsi, gt_hr_rgb in tqdm(dataloader):
            if path2save:
                path2save = Path(path2save)
            h_min, h_max, w_min, w_max = get_padding_coords(gt_hr_rgb)
            if inference:
                lr_hsi_patches = cut2patches(gt_lr_hsi)
                hr_rgb_patches = cut2patches(gt_hr_rgb)
                h, w = gt_hr_rgb.shape[2:]
            else:
                lr_hsi_patches = gt_lr_hsi
                hr_rgb_patches = gt_hr_rgb
            h, w = gt_hr_rgb.shape[2:]
            rec_hsi_patches = None
            gt_hr_hsi = gt_hr_hsi.to(device)
            gt_lr_hsi = gt_lr_hsi.to(device)
            if not inference:
                gt_hr_rgb = gt_hr_rgb.to(device)
            for batch_start in trange(0, len(hr_rgb_patches), batch_size, leave=False):
                lr_hsi = lr_hsi_patches[batch_start : batch_start + batch_size].to(
                    device
                )
                hr_rgb = hr_rgb_patches[batch_start : batch_start + batch_size].to(
                    device
                )
                with torch.no_grad():
                    rec_hr_hsi = model(lr_hsi, hr_rgb)
                if inference:
                    rec_hr_hsi = rec_hr_hsi.to("cpu")
                    gt_lr_hsi = gt_lr_hsi.to("cpu")
                if rec_hsi_patches is None:
                    rec_hsi_patches = rec_hr_hsi
                else:
                    rec_hsi_patches = torch.cat((rec_hsi_patches, rec_hr_hsi), dim=0)
            if inference:
                print("start merging patches")
            rec_hr_hsi = mergepathces(rec_hsi_patches, H=h, W=w)
            if inference:
                rec_hr_hsi = remove_padding(rec_hr_hsi, h_min, h_max, w_min, w_max)
            del lr_hsi_patches
            del hr_rgb_patches
            if interpolate:
                rec_hr_hsi = interpolate_tensor(rec_hr_hsi)
                gt_lr_hsi = interpolate_tensor(gt_lr_hsi)
            if inference:
                gt_lr_hsi = remove_padding(gt_lr_hsi)

            # h_min, h_max, w_min, w_max = get_padding_coords(gt_hr_hsi)
            # gt_hr_hsi = remove_padding(gt_hr_hsi)
            # gt_lr_hsi = remove_padding(gt_lr_hsi)
            # rec_hr_hsi = remove_padding(rec_hr_hsi, h_min, h_max, w_min, w_max)

            metrics_logger.compute_metrics(
                unstandartize_img(gt_lr_hsi, "hsi"),
                unstandartize_img(gt_hr_hsi, "hsi"),
                unstandartize_img(rec_hr_hsi, "hsi"),
            )
            if inference:
                print("metrics have been calculated")
            if path2save:
                rec_hr_hsi = rec_hr_hsi.cpu().numpy()[0].transpose(1, 2, 0)
                if hdr:
                    rec_hr_hsi = unstandartize_img(rec_hr_hsi, "hsi")
                    rec_hr_hsi = np.clip(rec_hr_hsi, 0, None)
                n_bands = rec_hr_hsi.shape[-1]
                if n_bands > 100:
                    rec_rgb = rec_hr_hsi[..., [70, 53, 19]]
                else:
                    rec_rgb = rec_hr_hsi[..., [20, 15, 5]]
                rec_rgb /= np.quantile(rec_rgb, 0.95)
                np.save(path2save / f"{imgs_names[i]}.npy", rec_hr_hsi)
                imwrite(path2save / f"{imgs_names[i]}.png", rec_rgb)
                if show:
                    h, w = gt_lr_hsi.shape[-2:]
                    rec_lr_hsi = cv2.resize(rec_hr_hsi, (w, h), cv2.INTER_AREA)
                    np.save(path2save / f"{imgs_names[i]}.npy", rec_lr_hsi)
                if inference:
                    print("img have been written")
                i += 1
            del rec_hr_hsi

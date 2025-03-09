import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from oml.retrieval import RetrievalResults, AdaptiveThresholding
from oml.metrics import calc_retrieval_metrics_rr
from oml.inference import inference
from tqdm import tqdm


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

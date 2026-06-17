# =============================================================================
# modules/prototype_classifier.py
# =============================================================================
# Few-shot por PROTOTIPOS (Euclidiana) sobre EasyFSL — version del clasificador
# que NO entrena: el backbone va 100% congelado y la "clasificacion" es por
# distancia al prototipo (la media de embeddings) de cada clase.
#
# DOS PROTOTIPOS:
#   Con el support set balanced (N glaucoma + N normal) se construye un prototipo
#   por clase. Una imagen nueva se asigna al prototipo mas cercano en distancia
#   Euclidiana (PrototypicalNetworks de Snell et al., via EasyFSL).
#
# DROP-IN DE CNNClassifier:
#   Expone la MISMA interfaz que modules/cnn_classifier.py para que el resto del
#   experimento (scripts/few_shot.py, scripts/evaluate.py, el modulo compartido
#   de Grad-CAM) funcione sin cambios:
#     - .model        : nn.Module (PrototypicalNetworks) que devuelve scores (B, C)
#     - .class_names  : ["normal", "glaucoma"]
#     - .device, .target_layer (ultima conv, para Grad-CAM)
#     - fit(support_loader)  -> dict        (reemplaza a train(): NO hay backprop)
#     - predict(image)       -> dict
#     - get_gradcam(image)   -> ndarray (H, W) en [0, 1]
#     - save(path) / load(path)
#
# NOTA — el score que devuelve self.model:
#   PrototypicalNetworks(use_softmax=False) -> forward devuelve la distancia
#   Euclidiana NEGATIVA a cada prototipo (mas cerca = score mayor). Asi argmax
#   elige el prototipo mas cercano y evaluate.py (que hace logits.argmax) funciona
#   sin tocarse. predict() aplica su propio softmax para la 'distribution'.
#
# ORDEN DE CLASES:
#   EasyFSL ordena los prototipos por label ascendente (labels.unique() ordena),
#   asi prototipos[0]=normal(0), prototipos[1]=glaucoma(1) -> casa con class_names
#   y con CLASS_TO_IDX de data_module.py (normal=0, glaucoma=1).
#
# GRAD-CAM CON BACKBONE CONGELADO:
#   Como NINGUN parametro entrena, el gradiente debe entrar por la IMAGEN
#   (batch.requires_grad_(True)); si no, las activaciones no tendrian grad. Se usa
#   el patron forward-hook + tensor.register_hook (igual que CNNClassifier), que
#   funciona con los 3 backbones (densenet aplica relu in-place sobre la salida de
#   `features`, lo que romperia un register_full_backward_hook).
#
# REQUIERE: torch, torchvision y easyfsl (pip install easyfsl) -> entorno Colab.
# =============================================================================

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from easyfsl.methods import PrototypicalNetworks

from modules.backbones import build_backbone
from modules.data_module import CLASS_TO_IDX, set_global_seed

logger = logging.getLogger(__name__)

# Nombres de clase por indice. DEBE casar con CLASS_TO_IDX de data_module.py
# (normal=0, glaucoma=1).
DEFAULT_CLASS_NAMES = ["normal", "glaucoma"]


class PrototypeClassifier:
    """
    Clasificador few-shot por prototipos Euclidianos (EasyFSL PrototypicalNetworks).

    Backbone pre-entrenado (ImageNet) CONGELADO -> embedding (B, D). El 'fit' solo
    calcula los prototipos (media por clase) del support set; no hay entrenamiento.
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: {backbone, [num_classes], [pretrained], [seed], [class_names]}.
        """
        self.backbone_name: str = config["backbone"]
        self.num_classes: int = config.get("num_classes", 2)
        self.pretrained: bool = config.get("pretrained", True)
        self.seed: int = config.get("seed", 42)
        # freeze_backbone=True -> prototipos sin entrenar; False -> backbone
        # entrenable para meta-training episodico (ver meta_train).
        self.freeze_backbone: bool = config.get("freeze_backbone", True)
        self.class_names: list[str] = config.get(
            "class_names",
            DEFAULT_CLASS_NAMES
            if self.num_classes == 2
            else [f"class_{i}" for i in range(self.num_classes)],
        )

        set_global_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Backbone + capa objetivo de Grad-CAM (la ultima conv) + su ruta string.
        backbone, self.target_layer, target_path, self.feature_dim = build_backbone(
            self.backbone_name, pretrained=self.pretrained, freeze=self.freeze_backbone
        )
        # Ruta de la capa objetivo para el modulo compartido de Grad-CAM. self.model
        # es el wrapper de EasyFSL, que guarda el backbone en .backbone -> prefijo.
        self.target_layer_path = f"backbone.{target_path}"

        # PrototypicalNetworks: distancia L2 a los prototipos. use_softmax=False
        # para que self.model devuelva scores crudos (distancia negativa), igual
        # que una CNN devuelve logits pre-softmax.
        self.model = PrototypicalNetworks(backbone, use_softmax=False).to(self.device)
        self.model.eval()

        logger.info(
            "PrototypeClassifier(%s) | euclidiana | freeze=%s | feat_dim=%d | device=%s",
            self.backbone_name,
            self.freeze_backbone,
            self.feature_dim,
            self.device,
        )

    # ---- Ajuste (reemplaza a train(): NO hay backprop) ----------------------

    def fit(self, support_loader) -> dict:
        """
        Calcula los prototipos (media de embeddings por clase) del support set.

        Recorre el support_loader (pocas imagenes: 2N <= 24), apila todo en un
        solo batch y llama a process_support_set de EasyFSL. Sin gradiente.

        Args:
            support_loader: DataLoader cuyos batches son dicts con "image"/"label"
                (formato de M1 DataModule).

        Returns:
            {"n_support": int, "n_classes_present": int, "prototype_shape": tuple}.
        """
        images: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for batch in support_loader:
            images.append(batch["image"])
            label = batch["label"]
            labels.append(label if torch.is_tensor(label) else torch.as_tensor(label))

        if not images:
            raise ValueError("El support_loader esta vacio: no hay con que crear prototipos.")

        support_images = torch.cat(images, dim=0).to(self.device)
        support_labels = torch.cat(labels, dim=0).to(self.device)

        n_classes_present = int(torch.unique(support_labels).numel())
        if n_classes_present < self.num_classes:
            logger.warning(
                "El support set tiene %d clase(s), se esperaban %d. Con 'balanced' "
                "deberian estar ambas (glaucoma y normal).",
                n_classes_present,
                self.num_classes,
            )

        self.model.eval()
        with torch.no_grad():
            self.model.process_support_set(support_images, support_labels)

        proto_shape = tuple(self.model.prototypes.shape)
        logger.info(
            "[%s] fit listo | support=%d | clases=%d | prototipos=%s",
            self.backbone_name,
            int(support_images.size(0)),
            n_classes_present,
            proto_shape,
        )
        return {
            "n_support": int(support_images.size(0)),
            "n_classes_present": n_classes_present,
            "prototype_shape": proto_shape,
        }

    # ---- Meta-training (episodic) -------------------------------------------

    def meta_train(
        self,
        data_module,
        *,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_episodes: int,
        lr: float,
        augment: bool = True,
    ) -> dict:
        """
        Meta-entrena (episodic training) el backbone con muestreo episodico propio.

        Por cada episodio muestrea n_way clases y, por clase, n_shot soporte +
        n_query query del split train; arma prototipos del support y minimiza
        CrossEntropy sobre el query -> backprop al backbone. Al terminar deja el
        modelo en eval(); los prototipos de despliegue los fija despues fit().

        Se muestrea a mano (sin el TaskSampler de EasyFSL, que es incompatible con
        algunas versiones de torch.utils.data.Sampler). Requiere freeze_backbone=False.

        Returns:
            {"n_episodes": int, "mean_loss": float}.
        """
        if self.freeze_backbone:
            logger.warning(
                "meta_train con freeze_backbone=True: el backbone esta congelado y "
                "no se actualizara. Construye con freeze_backbone=False."
            )

        train_ids = data_module.splits["train"]
        base = data_module.build_dataset(train_ids, augment=augment)
        # Indices del dataset agrupados por clase (label entero).
        items_per_label: dict[int, list[int]] = {}
        for idx, image_id in enumerate(train_ids):
            lbl = CLASS_TO_IDX[data_module.annotations[image_id]["label"]]
            items_per_label.setdefault(lbl, []).append(idx)

        classes = sorted(items_per_label)
        if len(classes) < n_way:
            raise ValueError(f"n_way={n_way} pero solo hay {len(classes)} clases en train.")

        rng = random.Random(self.seed)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        total_loss = 0.0

        for _ in range(n_episodes):
            # Muestrear n_way clases y, por cada una, n_shot soporte + n_query query.
            episode_classes = rng.sample(classes, n_way)
            s_imgs, s_lbls, q_imgs, q_lbls = [], [], [], []
            for new_label, c in enumerate(episode_classes):
                picks = rng.sample(items_per_label[c], n_shot + n_query)
                for j, idx in enumerate(picks):
                    img = base[idx]["image"]
                    if j < n_shot:
                        s_imgs.append(img)
                        s_lbls.append(new_label)
                    else:
                        q_imgs.append(img)
                        q_lbls.append(new_label)

            support_images = torch.stack(s_imgs).to(self.device)
            support_labels = torch.tensor(s_lbls, dtype=torch.long).to(self.device)
            query_images = torch.stack(q_imgs).to(self.device)
            query_labels = torch.tensor(q_lbls, dtype=torch.long).to(self.device)

            optimizer.zero_grad()
            self.model.process_support_set(support_images, support_labels)
            scores = self.model(query_images)
            loss = F.cross_entropy(scores, query_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        self.model.eval()
        mean_loss = total_loss / max(n_episodes, 1)
        logger.info(
            "[%s] meta_train listo | %d episodios | n_way=%d n_shot=%d n_query=%d | loss_media=%.4f",
            self.backbone_name,
            n_episodes,
            n_way,
            n_shot,
            n_query,
            mean_loss,
        )
        return {"n_episodes": n_episodes, "mean_loss": mean_loss}

    # ---- Inferencia ----------------------------------------------------------

    def predict(self, image: torch.Tensor) -> dict:
        """
        Clasifica una imagen ya normalizada por el prototipo mas cercano.

        Args:
            image: Tensor (3, H, W) o (1, 3, H, W).

        Returns:
            {"prediction": str, "distribution": {clase: prob}, "predicted_index": int}.
            La 'distribution' es el softmax de las distancias negativas (pseudo-prob).
        """
        self._check_fitted()
        self.model.eval()
        batch = self._ensure_batch(image).to(self.device)
        with torch.no_grad():
            scores = self.model(batch)  # (1, C) distancia Euclidiana negativa
            probs = F.softmax(scores, dim=1)[0].cpu().numpy()

        predicted_index = int(probs.argmax())
        distribution = {name: float(probs[i]) for i, name in enumerate(self.class_names)}
        return {
            "prediction": self.class_names[predicted_index],
            "distribution": distribution,
            "predicted_index": predicted_index,
        }

    def get_gradcam(self, image: torch.Tensor) -> np.ndarray:
        """
        Grad-CAM (heatmap continuo) del score del prototipo de la clase predicha.

        El backbone esta congelado, asi que se fuerza el gradiente por la imagen
        (batch.requires_grad_(True)). Devuelve un mapa (H, W) en [0, 1] SIN
        binarizar (la binarizacion + IoU vs GT las hace el evaluador de Grad-CAM).

        Args:
            image: Tensor (3, H, W) o (1, 3, H, W) normalizado.
        """
        self._check_fitted()
        self.model.eval()
        batch = self._ensure_batch(image).to(self.device)
        batch.requires_grad_(True)  # backbone congelado: el grad entra por la imagen

        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        def forward_hook(_module, _inp, output: torch.Tensor) -> None:
            activations["value"] = output
            if output.requires_grad:
                output.register_hook(
                    lambda grad: gradients.__setitem__("value", grad.detach())
                )

        handle = self.target_layer.register_forward_hook(forward_hook)
        try:
            scores = self.model(batch)  # (1, C)
            class_idx = int(scores.argmax(dim=1).item())
            self.model.zero_grad()
            scores[0, class_idx].backward()

            if "value" not in gradients:
                raise RuntimeError(
                    "No se capturo gradiente en la capa objetivo. Verifica que "
                    "batch.requires_grad_(True) y que target_layer participa en el forward."
                )

            acts = activations["value"].detach()  # (1, C, h, w)
            grads = gradients["value"]  # (1, C, h, w)
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (1, 1, h, w)
            cam = F.interpolate(
                cam, size=batch.shape[-2:], mode="bilinear", align_corners=False
            )[0, 0]
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            return cam.detach().cpu().numpy()
        finally:
            handle.remove()

    # ---- Persistencia --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Guarda los prototipos + metadatos. Si el backbone fue meta-entrenado
        (freeze_backbone=False) tambien guarda sus pesos; si esta congelado es
        ImageNet y se reconstruye en __init__ (no hace falta guardarlo).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        prototypes = getattr(self.model, "prototypes", None)
        state = {
            "method": "euclidean_prototypes" if self.freeze_backbone else "meta_prototypes",
            "backbone": self.backbone_name,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "prototypes": prototypes.detach().cpu() if prototypes is not None else None,
        }
        if not self.freeze_backbone:
            # Backbone meta-entrenado: hay que persistir sus pesos para recargarlo.
            state["backbone_state"] = self.model.state_dict()
        torch.save(state, path)
        logger.info("Prototipos guardados en %s", path)

    def load(self, path: str | Path) -> None:
        """
        Carga prototipos (y, si el modelo fue meta-entrenado, los pesos del
        backbone). El backbone base ya esta construido en __init__.
        """
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint {path} con formato inesperado.")
        backbone_state = checkpoint.get("backbone_state")
        if backbone_state is not None:
            # Modelo meta-entrenado: restaurar los pesos del backbone entrenado.
            self.model.load_state_dict(backbone_state)
            self.model.to(self.device)
        prototypes = checkpoint.get("prototypes")
        if prototypes is None:
            raise ValueError(f"El checkpoint {path} no contiene 'prototypes'.")
        self.model.prototypes = prototypes.to(self.device)
        logger.info("Prototipos cargados desde %s", path)

    # ---- Helpers -------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Verifica que ya se llamo a fit() (prototipos disponibles)."""
        if getattr(self.model, "prototypes", None) is None:
            raise RuntimeError("Llama a fit(support_loader) antes de predecir o sacar Grad-CAM.")

    @staticmethod
    def _ensure_batch(image: torch.Tensor) -> torch.Tensor:
        """Asegura forma (1, 3, H, W) a partir de (3, H, W) o (1, 3, H, W)."""
        if image.dim() == 3:
            return image.unsqueeze(0)
        if image.dim() == 4:
            return image
        raise ValueError(f"Imagen con forma inesperada: {tuple(image.shape)}")

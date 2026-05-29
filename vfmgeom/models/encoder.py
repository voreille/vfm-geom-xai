import timm
import torch
import torch.nn as nn


class Uni2EncoderSimple(nn.Module):
    def __init__(
        self,
        encoder_name: str = "hf-hub:MahmoodLab/UNI2-h",
        img_size: tuple[int, int] = (448, 448),
        ckpt_path: str = "",
        sub_norm: bool = False,
        patch_size: int = 14,
        pretrained: bool = True,
    ):
        super().__init__()

        model_kwargs = {
            "model_name": encoder_name,
            "pretrained": pretrained,
        }
        if patch_size != 14:
            raise ValueError("Uni2 only supports patch size of 14")

        timm_kwargs = {
            "img_size": 224,
            "patch_size": patch_size,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        model_kwargs.update(timm_kwargs)
        self.encoder = timm.create_model(**model_kwargs)

        pixel_mean = torch.tensor(self.encoder.default_cfg["mean"]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(self.encoder.default_cfg["std"]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        self.grid_size = tuple(round(size / patch_size) for size in img_size)

        self.embed_dim = (
            self.encoder.embed_dim
            if hasattr(self.encoder, "embed_dim")
            else self.encoder.num_features
        )

    def forward(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        x = self.encoder.forward_features(x)
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)
        else:
            x = x[:, self.encoder.num_prefix_tokens :]
        return x


def build_encoder(encoder_id: str) -> tuple[nn.Module, dict]:
    if encoder_id == "uni2h":
        amp_dtype = torch.bfloat16
        timm_kwargs = {
            "model_name": "hf-hub:MahmoodLab/UNI2-h",
            "pretrained": True,
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        encoder = timm.create_model(**timm_kwargs)

        embed_dim = 1536
        patch_size = 14
        pixel_mean = encoder.default_cfg["mean"]
        pixel_std = encoder.default_cfg["std"]
        n_blocks = len(encoder.blocks)
    elif encoder_id == "h-optimus-1":
        amp_dtype = torch.float16
        encoder = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        embed_dim = 1536
        patch_size = 14
        pixel_mean = [0.707223, 0.578729, 0.703617]
        pixel_std = [0.211883, 0.230117, 0.177517]
        n_blocks = len(encoder.blocks)
    elif encoder_id == "h0-mini":
        amp_dtype = torch.float16
        encoder = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            dynamic_img_size=True,  # keep this so your hooks work on 448
        )
        embed_dim = getattr(encoder, "embed_dim", 768)
        patch_size = 14
        pixel_mean = encoder.default_cfg[
            "mean"
        ]  # I checked these are the same as h-optimus-1
        pixel_std = encoder.default_cfg["std"]
        n_blocks = len(encoder.blocks)
    elif encoder_id == "vit-small":
        encoder = timm.create_model(
            "vit_small_patch16_224.augreg_in21k", pretrained=True, num_classes=0
        )
        amp_dtype = torch.float16
        embed_dim = getattr(encoder, "embed_dim")
        patch_size = 16
        pixel_mean = encoder.default_cfg[
            "mean"
        ]  # I checked these are the same as h-optimus-1
        pixel_std = encoder.default_cfg["std"]
        n_blocks = len(encoder.blocks)

    else:
        raise ValueError(f"unknown encoder_id {encoder_id}")

    return encoder, {
        "amp_dtype": amp_dtype,
        "embed_dim": embed_dim,
        "patch_size": patch_size,
        "pixel_mean": pixel_mean,
        "pixel_std": pixel_std,
        "n_blocks": n_blocks,
    }

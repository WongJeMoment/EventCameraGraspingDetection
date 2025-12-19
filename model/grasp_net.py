# grasp_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import build_backbone, MobileNetV3Small, ShuffleNetV2
from model.decoder import LiteFPNDecoder
from model.heads import GraspHead


class GraspNet(nn.Module):
    """
    Backbone + Decoder(Neck) + Head

    Outputs:
      Q_logits: (B,1,Hm,Wm)   Hm=H/stride (default stride=4)
      reg:      (B,5,Hm,Wm)   dx,dy,logw,sin,cos

    Notes:
      - Q_logits should be passed through sigmoid to get qmap probability.
      - reg is already constrained if head options enabled (dxdy sigmoid, sincos norm, logw clamp).
    """
    def __init__(
        self,
        backbone_name="mobilenetv3_small",
        backbone_width=1.0,
        neck_out_ch=96,
        neck_num_dw=1,
        head_ch=96,
        head_num_dw=1,
        stride=4,
        normalize_sincos=True,
        dxdy_sigmoid=True,
        logw_clamp=(-5.0, 6.0),
    ):
        super().__init__()
        assert stride in (4, 8), "This implementation assumes final stride is 4 (c2) or 8 (c3)."

        self.stride = stride
        self.backbone_name = backbone_name

        # ---- backbone ----
        self.backbone = build_backbone(backbone_name, width_mult=backbone_width)

        # ---- infer channel sizes automatically (safe) ----
        # we run a tiny dummy forward once to get channel dims
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 240, 320)
            feats = self.backbone(dummy)
            c2_ch = feats["c2"].shape[1]
            c3_ch = feats["c3"].shape[1]
            c4_ch = feats["c4"].shape[1]

        # ---- neck/decoder (lite FPN) -> produce stride-4 feature (p2) ----
        self.neck = LiteFPNDecoder(
            c2_ch=c2_ch, c3_ch=c3_ch, c4_ch=c4_ch,
            out_ch=neck_out_ch,
            num_dw=neck_num_dw,
            reg_norm=False,  # sincos norm should be handled in head to be consistent
        )

        # ---- head ----
        self.head = GraspHead(
            in_ch=neck_out_ch,
            head_ch=head_ch,
            num_dw=head_num_dw,
            normalize_sincos=normalize_sincos,
            dxdy_sigmoid=dxdy_sigmoid,
            logw_clamp=logw_clamp,
        )

    def forward(self, x):
        feats = self.backbone(x)                 # dict c2,c3,c4
        neck_feat, _ = self._neck_feat(feats)    # stride-4 feature
        Q_logits, reg = self.head(neck_feat)
        return Q_logits, reg

    def _neck_feat(self, feats):
        """
        Use LiteFPNDecoder to get:
          p2 feature (stride 4), and intermediate outputs (optional)
        We reuse decoder forward but expose its p2 feature by small modification:
        here we call decoder and treat its p2 as the internal feature before heads.
        """
        # LiteFPNDecoder.forward returns (Q, reg) in the earlier code I gave.
        # But we want its final feature map p2 for our GraspHead.
        # So we implement a lightweight extraction path here:
        c2, c3, c4 = feats["c2"], feats["c3"], feats["c4"]

        # these layers exist inside LiteFPNDecoder
        lat2, lat3, lat4 = self.neck.lat2, self.neck.lat3, self.neck.lat4
        refine2, refine3, refine4 = self.neck.refine2, self.neck.refine3, self.neck.refine4

        p4 = refine4(lat4(c4))  # s16
        p3 = lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = refine3(p3)        # s8
        p2 = lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = refine2(p2)        # s4
        return p2, {"p3": p3, "p4": p4}

    @torch.no_grad()
    def infer(self, x, return_numpy=True):
        """
        Inference helper:
          returns qmap_prob and reg maps.
        """
        self.eval()
        Q_logits, reg = self.forward(x)
        qmap = torch.sigmoid(Q_logits)

        if return_numpy:
            qmap_np = qmap.squeeze(1).cpu().numpy()     # (B,Hm,Wm)
            reg_np = reg.cpu().numpy()                  # (B,5,Hm,Wm)
            return qmap_np, reg_np
        return qmap, reg


if __name__ == "__main__":
    # quick sanity
    net = GraspNet(backbone_name="mobilenetv3_small", backbone_width=1.0)
    x = torch.randn(1, 3, 240, 320)
    Q, reg = net(x)
    print("Q:", Q.shape, "reg:", reg.shape)  # (1,1,60,80) (1,5,60,80)

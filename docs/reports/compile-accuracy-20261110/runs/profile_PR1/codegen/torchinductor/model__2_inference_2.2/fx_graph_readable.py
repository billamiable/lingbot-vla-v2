class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[96, 3]"):
         # File: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:229 in get_image_features, code: image_grid_cpu = image_grid_thw.detach().cpu()
        device_put: "i64[96, 3]" = torch.ops.prims.device_put.default(arg0_1, device(type='cpu'));  arg0_1 = None
        return (device_put,)
        
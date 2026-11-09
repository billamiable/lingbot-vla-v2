


def forward(self, L_z_ : torch.Tensor, L_t_ : torch.Tensor, G_a_modules_self_attn_modules_o_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_a_modules_post_attention_layernorm_parameters_weight_ : torch.nn.parameter.Parameter, G_a_modules_mlp_modules_gate_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_a_modules_mlp_modules_up_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_a_modules_mlp_modules_down_proj_parameters_weight_ : torch.nn.parameter.Parameter):
    l_z_ = L_z_
    l_t_ = L_t_
    g_a_modules_self_attn_modules_o_proj_parameters_weight_ = G_a_modules_self_attn_modules_o_proj_parameters_weight_
    g_a_modules_post_attention_layernorm_parameters_weight_ = G_a_modules_post_attention_layernorm_parameters_weight_
    g_a_modules_mlp_modules_gate_proj_parameters_weight_ = G_a_modules_mlp_modules_gate_proj_parameters_weight_
    g_a_modules_mlp_modules_up_proj_parameters_weight_ = G_a_modules_mlp_modules_up_proj_parameters_weight_
    g_a_modules_mlp_modules_down_proj_parameters_weight_ = G_a_modules_mlp_modules_down_proj_parameters_weight_
    wrap_body_0 = self.wrap_body_0
    tag_activation_checkpoint = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_0, l_z_, l_t_, g_a_modules_self_attn_modules_o_proj_parameters_weight_, g_a_modules_post_attention_layernorm_parameters_weight_, g_a_modules_mlp_modules_gate_proj_parameters_weight_, g_a_modules_mlp_modules_up_proj_parameters_weight_, g_a_modules_mlp_modules_down_proj_parameters_weight_, use_reentrant = False);  wrap_body_0 = l_z_ = l_t_ = g_a_modules_self_attn_modules_o_proj_parameters_weight_ = g_a_modules_post_attention_layernorm_parameters_weight_ = g_a_modules_mlp_modules_gate_proj_parameters_weight_ = g_a_modules_mlp_modules_up_proj_parameters_weight_ = g_a_modules_mlp_modules_down_proj_parameters_weight_ = None
    getitem = tag_activation_checkpoint[0];  tag_activation_checkpoint = None
    return (getitem,)
    



def forward(self, L_z_ : torch.Tensor, L_t_ : torch.Tensor, G_v_modules_self_attn_modules_o_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_v_modules_post_attention_layernorm_parameters_weight_ : torch.nn.parameter.Parameter, G_v_modules_mlp_modules_gate_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_v_modules_mlp_modules_up_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_v_modules_mlp_modules_down_proj_parameters_weight_ : torch.nn.parameter.Parameter):
    l_z_ = L_z_
    l_t_ = L_t_
    g_v_modules_self_attn_modules_o_proj_parameters_weight_ = G_v_modules_self_attn_modules_o_proj_parameters_weight_
    g_v_modules_post_attention_layernorm_parameters_weight_ = G_v_modules_post_attention_layernorm_parameters_weight_
    g_v_modules_mlp_modules_gate_proj_parameters_weight_ = G_v_modules_mlp_modules_gate_proj_parameters_weight_
    g_v_modules_mlp_modules_up_proj_parameters_weight_ = G_v_modules_mlp_modules_up_proj_parameters_weight_
    g_v_modules_mlp_modules_down_proj_parameters_weight_ = G_v_modules_mlp_modules_down_proj_parameters_weight_
    hidden_states = l_z_.to(torch.bfloat16);  l_z_ = None
    att_output = l_t_.to(torch.bfloat16);  l_t_ = None
    getitem = att_output[(slice(None, None, None), slice(0, 8, None))];  att_output = None
    out_emb = torch._C._nn.linear(getitem, g_v_modules_self_attn_modules_o_proj_parameters_weight_, None);  getitem = g_v_modules_self_attn_modules_o_proj_parameters_weight_ = None
    out_emb += hidden_states;  out_emb_1 = out_emb;  out_emb = hidden_states = None
    after_first_residual = out_emb_1.clone()
    hidden_states_1 = out_emb_1.to(torch.float32);  out_emb_1 = None
    pow_1 = hidden_states_1.pow(2)
    variance = pow_1.mean(-1, keepdim = True);  pow_1 = None
    add = variance + 1e-06;  variance = None
    rsqrt = torch.rsqrt(add);  add = None
    hidden_states_2 = hidden_states_1 * rsqrt;  hidden_states_1 = rsqrt = None
    to_3 = hidden_states_2.to(torch.bfloat16);  hidden_states_2 = None
    out_emb_2 = g_v_modules_post_attention_layernorm_parameters_weight_ * to_3;  g_v_modules_post_attention_layernorm_parameters_weight_ = to_3 = None
    linear_1 = torch._C._nn.linear(out_emb_2, g_v_modules_mlp_modules_gate_proj_parameters_weight_, None);  g_v_modules_mlp_modules_gate_proj_parameters_weight_ = None
    silu = torch.nn.functional.silu(linear_1);  linear_1 = None
    linear_2 = torch._C._nn.linear(out_emb_2, g_v_modules_mlp_modules_up_proj_parameters_weight_, None);  out_emb_2 = g_v_modules_mlp_modules_up_proj_parameters_weight_ = None
    mul_2 = silu * linear_2;  silu = linear_2 = None
    down_proj = torch._C._nn.linear(mul_2, g_v_modules_mlp_modules_down_proj_parameters_weight_, None);  mul_2 = g_v_modules_mlp_modules_down_proj_parameters_weight_ = None
    down_proj += after_first_residual;  out_emb_3 = down_proj;  down_proj = after_first_residual = None
    return (out_emb_3,)
    
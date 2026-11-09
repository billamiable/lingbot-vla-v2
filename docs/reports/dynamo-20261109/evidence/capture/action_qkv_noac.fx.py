


def forward(self, G_a_modules_self_attn_modules_q_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_z_ : torch.Tensor, G_a_modules_input_layernorm_parameters_weight_ : torch.nn.parameter.Parameter, G_a_modules_self_attn_modules_q_proj_parameters_bias_ : torch.nn.parameter.Parameter, G_a_modules_self_attn_modules_k_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_a_modules_self_attn_modules_k_proj_parameters_bias_ : torch.nn.parameter.Parameter, G_a_modules_self_attn_modules_v_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_a_modules_self_attn_modules_v_proj_parameters_bias_ : torch.nn.parameter.Parameter):
    g_a_modules_self_attn_modules_q_proj_parameters_weight_ = G_a_modules_self_attn_modules_q_proj_parameters_weight_
    l_z_ = L_z_
    g_a_modules_input_layernorm_parameters_weight_ = G_a_modules_input_layernorm_parameters_weight_
    g_a_modules_self_attn_modules_q_proj_parameters_bias_ = G_a_modules_self_attn_modules_q_proj_parameters_bias_
    g_a_modules_self_attn_modules_k_proj_parameters_weight_ = G_a_modules_self_attn_modules_k_proj_parameters_weight_
    g_a_modules_self_attn_modules_k_proj_parameters_bias_ = G_a_modules_self_attn_modules_k_proj_parameters_bias_
    g_a_modules_self_attn_modules_v_proj_parameters_weight_ = G_a_modules_self_attn_modules_v_proj_parameters_weight_
    g_a_modules_self_attn_modules_v_proj_parameters_bias_ = G_a_modules_self_attn_modules_v_proj_parameters_bias_
    hidden_states = l_z_.to(torch.float32);  l_z_ = None
    hidden_states_1 = hidden_states.to(torch.float32);  hidden_states = None
    pow_1 = hidden_states_1.pow(2)
    variance = pow_1.mean(-1, keepdim = True);  pow_1 = None
    add = variance + 1e-06;  variance = None
    rsqrt = torch.rsqrt(add);  add = None
    hidden_states_2 = hidden_states_1 * rsqrt;  hidden_states_1 = rsqrt = None
    to_2 = hidden_states_2.to(torch.float32);  hidden_states_2 = None
    hidden_states_3 = g_a_modules_input_layernorm_parameters_weight_ * to_2;  g_a_modules_input_layernorm_parameters_weight_ = to_2 = None
    linear = torch._C._nn.linear(hidden_states_3, g_a_modules_self_attn_modules_q_proj_parameters_weight_, g_a_modules_self_attn_modules_q_proj_parameters_bias_);  g_a_modules_self_attn_modules_q_proj_parameters_weight_ = g_a_modules_self_attn_modules_q_proj_parameters_bias_ = None
    query_state = linear.view((1, 4, -1, 128));  linear = None
    linear_1 = torch._C._nn.linear(hidden_states_3, g_a_modules_self_attn_modules_k_proj_parameters_weight_, g_a_modules_self_attn_modules_k_proj_parameters_bias_);  g_a_modules_self_attn_modules_k_proj_parameters_weight_ = g_a_modules_self_attn_modules_k_proj_parameters_bias_ = None
    key_state = linear_1.view((1, 4, -1, 128));  linear_1 = None
    linear_2 = torch._C._nn.linear(hidden_states_3, g_a_modules_self_attn_modules_v_proj_parameters_weight_, g_a_modules_self_attn_modules_v_proj_parameters_bias_);  hidden_states_3 = g_a_modules_self_attn_modules_v_proj_parameters_weight_ = g_a_modules_self_attn_modules_v_proj_parameters_bias_ = None
    value_state = linear_2.view((1, 4, -1, 128));  linear_2 = None
    return (query_state, key_state, value_state)
    
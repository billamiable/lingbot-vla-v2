


def forward(self, L_self_modules_self_attn_modules_q_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_hidden_states_ : torch.Tensor, L_self_modules_input_layernorm_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_self_attn_modules_q_norm_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_self_attn_modules_k_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_self_attn_modules_k_norm_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_self_attn_modules_v_proj_parameters_weight_ : torch.nn.parameter.Parameter):
    l_self_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_self_attn_modules_q_proj_parameters_weight_
    l_hidden_states_ = L_hidden_states_
    l_self_modules_input_layernorm_parameters_weight_ = L_self_modules_input_layernorm_parameters_weight_
    l_self_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_self_attn_modules_q_norm_parameters_weight_
    l_self_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_self_attn_modules_k_proj_parameters_weight_
    l_self_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_self_attn_modules_k_norm_parameters_weight_
    l_self_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_self_attn_modules_v_proj_parameters_weight_
    hidden_states = l_hidden_states_.to(torch.bfloat16);  l_hidden_states_ = None
    hidden_states_1 = hidden_states.to(torch.float32);  hidden_states = None
    pow_1 = hidden_states_1.pow(2)
    variance = pow_1.mean(-1, keepdim = True);  pow_1 = None
    add = variance + 1e-06;  variance = None
    rsqrt = torch.rsqrt(add);  add = None
    hidden_states_2 = hidden_states_1 * rsqrt;  hidden_states_1 = rsqrt = None
    to_2 = hidden_states_2.to(torch.bfloat16);  hidden_states_2 = None
    hidden_states_3 = l_self_modules_input_layernorm_parameters_weight_ * to_2;  l_self_modules_input_layernorm_parameters_weight_ = to_2 = None
    linear = torch._C._nn.linear(hidden_states_3, l_self_modules_self_attn_modules_q_proj_parameters_weight_, None);  l_self_modules_self_attn_modules_q_proj_parameters_weight_ = None
    view = linear.view((1, 8, -1, 128));  linear = None
    hidden_states_4 = view.to(torch.float32);  view = None
    pow_2 = hidden_states_4.pow(2)
    variance_1 = pow_2.mean(-1, keepdim = True);  pow_2 = None
    add_1 = variance_1 + 1e-06;  variance_1 = None
    rsqrt_1 = torch.rsqrt(add_1);  add_1 = None
    hidden_states_5 = hidden_states_4 * rsqrt_1;  hidden_states_4 = rsqrt_1 = None
    to_4 = hidden_states_5.to(torch.bfloat16);  hidden_states_5 = None
    query_state = l_self_modules_self_attn_modules_q_norm_parameters_weight_ * to_4;  l_self_modules_self_attn_modules_q_norm_parameters_weight_ = to_4 = None
    linear_1 = torch._C._nn.linear(hidden_states_3, l_self_modules_self_attn_modules_k_proj_parameters_weight_, None);  l_self_modules_self_attn_modules_k_proj_parameters_weight_ = None
    view_1 = linear_1.view((1, 8, -1, 128));  linear_1 = None
    hidden_states_6 = view_1.to(torch.float32);  view_1 = None
    pow_3 = hidden_states_6.pow(2)
    variance_2 = pow_3.mean(-1, keepdim = True);  pow_3 = None
    add_2 = variance_2 + 1e-06;  variance_2 = None
    rsqrt_2 = torch.rsqrt(add_2);  add_2 = None
    hidden_states_7 = hidden_states_6 * rsqrt_2;  hidden_states_6 = rsqrt_2 = None
    to_6 = hidden_states_7.to(torch.bfloat16);  hidden_states_7 = None
    key_state = l_self_modules_self_attn_modules_k_norm_parameters_weight_ * to_6;  l_self_modules_self_attn_modules_k_norm_parameters_weight_ = to_6 = None
    linear_2 = torch._C._nn.linear(hidden_states_3, l_self_modules_self_attn_modules_v_proj_parameters_weight_, None);  hidden_states_3 = l_self_modules_self_attn_modules_v_proj_parameters_weight_ = None
    value_state = linear_2.view((1, 8, -1, 128));  linear_2 = None
    return (query_state, key_state, value_state)
    
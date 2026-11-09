


def forward(self, L_z_ : torch.Tensor, G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_w_ : torch.Tensor, G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ : torch.nn.parameter.Parameter, G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ : torch.nn.parameter.Parameter, G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ : torch.nn.parameter.Parameter, G_position_ : torch.Tensor, G_j_modules_qwenvl_modules_model_modules_language_model_modules_rotary_emb_buffers_inv_freq_ : torch.Tensor, G_prepared_kv_indices : torch.Tensor, G_prepared_kv_num_blocks : torch.Tensor, G_prepared_mask_mod_closure_0_cell_contents : torch.Tensor, G_prepared_full_kv_num_blocks : torch.Tensor, G_prepared_full_kv_indices : torch.Tensor, G_prepared_q_num_blocks : torch.Tensor, G_prepared_q_indices : torch.Tensor, G_prepared_full_q_num_blocks : torch.Tensor, G_prepared_full_q_indices : torch.Tensor, G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ : torch.nn.parameter.Parameter):
    l_z_ = L_z_
    g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
    g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
    g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_ = G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_
    g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
    g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_ = G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_
    g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
    l_w_ = L_w_
    g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
    g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
    g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
    g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
    g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
    g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
    g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
    g_position_ = G_position_
    g_j_modules_qwenvl_modules_model_modules_language_model_modules_rotary_emb_buffers_inv_freq_ = G_j_modules_qwenvl_modules_model_modules_language_model_modules_rotary_emb_buffers_inv_freq_
    g_prepared_kv_indices = G_prepared_kv_indices
    g_prepared_kv_num_blocks = G_prepared_kv_num_blocks
    g_prepared_mask_mod_closure_0_cell_contents = G_prepared_mask_mod_closure_0_cell_contents
    g_prepared_full_kv_num_blocks = G_prepared_full_kv_num_blocks
    g_prepared_full_kv_indices = G_prepared_full_kv_indices
    g_prepared_q_num_blocks = G_prepared_q_num_blocks
    g_prepared_q_indices = G_prepared_q_indices
    g_prepared_full_q_num_blocks = G_prepared_full_q_num_blocks
    g_prepared_full_q_indices = G_prepared_full_q_indices
    g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
    g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
    g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_
    g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_
    g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = G_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_
    g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
    g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
    g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_
    g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_
    g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = G_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_
    hidden_states = l_z_.to(torch.bfloat16)
    hidden_states_1 = hidden_states.to(torch.float32);  hidden_states = None
    pow_1 = hidden_states_1.pow(2)
    variance = pow_1.mean(-1, keepdim = True);  pow_1 = None
    add = variance + 1e-06;  variance = None
    rsqrt = torch.rsqrt(add);  add = None
    hidden_states_2 = hidden_states_1 * rsqrt;  hidden_states_1 = rsqrt = None
    to_2 = hidden_states_2.to(torch.bfloat16);  hidden_states_2 = None
    hidden_states_3 = g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ * to_2;  g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = to_2 = None
    linear = torch._C._nn.linear(hidden_states_3, g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_, None);  g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = None
    view = linear.view((1, 8, -1, 128));  linear = None
    hidden_states_4 = view.to(torch.float32);  view = None
    pow_2 = hidden_states_4.pow(2)
    variance_1 = pow_2.mean(-1, keepdim = True);  pow_2 = None
    add_1 = variance_1 + 1e-06;  variance_1 = None
    rsqrt_1 = torch.rsqrt(add_1);  add_1 = None
    hidden_states_5 = hidden_states_4 * rsqrt_1;  hidden_states_4 = rsqrt_1 = None
    to_4 = hidden_states_5.to(torch.bfloat16);  hidden_states_5 = None
    query_state = g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_ * to_4;  g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_ = to_4 = None
    linear_1 = torch._C._nn.linear(hidden_states_3, g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_, None);  g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = None
    view_1 = linear_1.view((1, 8, -1, 128));  linear_1 = None
    hidden_states_6 = view_1.to(torch.float32);  view_1 = None
    pow_3 = hidden_states_6.pow(2)
    variance_2 = pow_3.mean(-1, keepdim = True);  pow_3 = None
    add_2 = variance_2 + 1e-06;  variance_2 = None
    rsqrt_2 = torch.rsqrt(add_2);  add_2 = None
    hidden_states_7 = hidden_states_6 * rsqrt_2;  hidden_states_6 = rsqrt_2 = None
    to_6 = hidden_states_7.to(torch.bfloat16);  hidden_states_7 = None
    key_state = g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_ * to_6;  g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_ = to_6 = None
    linear_2 = torch._C._nn.linear(hidden_states_3, g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_, None);  hidden_states_3 = g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = None
    value_state = linear_2.view((1, 8, -1, 128));  linear_2 = None
    float_1 = query_state.float();  query_state = None
    float_2 = key_state.float();  key_state = None
    float_3 = value_state.float();  value_state = None
    hidden_states_8 = l_w_.to(torch.float32)
    hidden_states_9 = hidden_states_8.to(torch.float32);  hidden_states_8 = None
    pow_4 = hidden_states_9.pow(2)
    variance_3 = pow_4.mean(-1, keepdim = True);  pow_4 = None
    add_3 = variance_3 + 1e-06;  variance_3 = None
    rsqrt_3 = torch.rsqrt(add_3);  add_3 = None
    hidden_states_10 = hidden_states_9 * rsqrt_3;  hidden_states_9 = rsqrt_3 = None
    to_9 = hidden_states_10.to(torch.float32);  hidden_states_10 = None
    hidden_states_11 = g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ * to_9;  g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = to_9 = None
    linear_3 = torch._C._nn.linear(hidden_states_11, g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_, g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_);  g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = None
    query_state_1 = linear_3.view((1, 4, -1, 128));  linear_3 = None
    linear_4 = torch._C._nn.linear(hidden_states_11, g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_, g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_);  g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = None
    key_state_1 = linear_4.view((1, 4, -1, 128));  linear_4 = None
    linear_5 = torch._C._nn.linear(hidden_states_11, g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_, g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_);  hidden_states_11 = g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = None
    value_state_1 = linear_5.view((1, 4, -1, 128));  linear_5 = None
    float_4 = query_state_1.float();  query_state_1 = None
    float_5 = key_state_1.float();  key_state_1 = None
    float_6 = value_state_1.float();  value_state_1 = None
    query_states = torch.cat([float_1, float_4], dim = 1);  float_1 = float_4 = None
    key_states = torch.cat([float_2, float_5], dim = 1);  float_2 = float_5 = None
    value_states = torch.cat([float_3, float_6], dim = 1);  float_3 = float_6 = None
    _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None
    getitem = g_j_modules_qwenvl_modules_model_modules_language_model_modules_rotary_emb_buffers_inv_freq_[(None, None, slice(None, None, None), None)];  g_j_modules_qwenvl_modules_model_modules_language_model_modules_rotary_emb_buffers_inv_freq_ = None
    float_7 = getitem.float();  getitem = None
    inv_freq_expanded = float_7.expand(3, 1, -1, 1);  float_7 = None
    getitem_1 = g_position_[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  g_position_ = None
    position_ids_expanded = getitem_1.float();  getitem_1 = None
    _enter_autocast = torch.amp.autocast_mode._enter_autocast('cuda', None, False, None)
    float_9 = inv_freq_expanded.float();  inv_freq_expanded = None
    float_10 = position_ids_expanded.float();  position_ids_expanded = None
    matmul = float_9 @ float_10;  float_9 = float_10 = None
    freqs = matmul.transpose(2, 3);  matmul = None
    freqs_t = freqs[0]
    getitem_3 = freqs[(1, Ellipsis, slice(1, 60, 3))]
    freqs_t[(Ellipsis, slice(1, 60, 3))] = getitem_3;  setitem = freqs_t;  getitem_3 = setitem = None
    getitem_4 = freqs[(2, Ellipsis, slice(2, 60, 3))];  freqs = None
    freqs_t[(Ellipsis, slice(2, 60, 3))] = getitem_4;  setitem_1 = freqs_t;  getitem_4 = setitem_1 = None
    emb = torch.cat((freqs_t, freqs_t), dim = -1);  freqs_t = None
    cos = emb.cos()
    cos_1 = cos * 1.0;  cos = None
    sin = emb.sin();  emb = None
    sin_1 = sin * 1.0;  sin = None
    _exit_autocast = torch.amp.autocast_mode._exit_autocast(_enter_autocast);  _enter_autocast = _exit_autocast = None
    to_10 = cos_1.to(dtype = torch.float32);  cos_1 = None
    to_11 = sin_1.to(dtype = torch.float32);  sin_1 = None
    _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
    cos_2 = to_10.unsqueeze(2);  to_10 = None
    sin_2 = to_11.unsqueeze(2);  to_11 = None
    mul_10 = query_states * cos_2
    x1 = query_states[(Ellipsis, slice(None, 64, None))]
    x2 = query_states[(Ellipsis, slice(64, None, None))];  query_states = None
    neg = -x2;  x2 = None
    cat_4 = torch.cat((neg, x1), dim = -1);  neg = x1 = None
    mul_11 = cat_4 * sin_2;  cat_4 = None
    q_embed = mul_10 + mul_11;  mul_10 = mul_11 = None
    mul_12 = key_states * cos_2;  cos_2 = None
    x1_1 = key_states[(Ellipsis, slice(None, 64, None))]
    x2_1 = key_states[(Ellipsis, slice(64, None, None))];  key_states = None
    neg_1 = -x2_1;  x2_1 = None
    cat_5 = torch.cat((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None
    mul_13 = cat_5 * sin_2;  cat_5 = sin_2 = None
    k_embed = mul_12 + mul_13;  mul_12 = mul_13 = None
    transpose_1 = q_embed.transpose(1, 2);  q_embed = None
    query_states_1 = transpose_1.to(torch.float32);  transpose_1 = None
    transpose_2 = k_embed.transpose(1, 2);  k_embed = None
    key_states_1 = transpose_2.to(torch.float32);  transpose_2 = None
    transpose_3 = value_states.transpose(1, 2);  value_states = None
    value_states_1 = transpose_3.to(torch.float32);  transpose_3 = None
    query_states_2 = torch._C._nn.pad(query_states_1, (0, 0, 0, 116), 'constant', 0.0);  query_states_1 = None
    key_states_2 = torch._C._nn.pad(key_states_1, (0, 0, 0, 116), 'constant', 0.0);  key_states_1 = None
    value_states_2 = torch._C._nn.pad(value_states_1, (0, 0, 0, 116), 'constant', 0.0);  value_states_1 = None
    score_mod_0 = self.score_mod_0
    mask_fn_0 = self.mask_fn_0
    flex_attention = torch.ops.higher_order.flex_attention(query_states_2, key_states_2, value_states_2, score_mod_0, (128, 128, g_prepared_kv_num_blocks, g_prepared_kv_indices, g_prepared_full_kv_num_blocks, g_prepared_full_kv_indices, g_prepared_q_num_blocks, g_prepared_q_indices, g_prepared_full_q_num_blocks, g_prepared_full_q_indices, 128, 128, mask_fn_0), 0.08838834764831845, {'BLOCK_M': 32, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2, 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True}, (), (g_prepared_mask_mod_closure_0_cell_contents,));  query_states_2 = key_states_2 = value_states_2 = score_mod_0 = g_prepared_kv_num_blocks = g_prepared_kv_indices = g_prepared_full_kv_num_blocks = g_prepared_full_kv_indices = g_prepared_q_num_blocks = g_prepared_q_indices = g_prepared_full_q_num_blocks = g_prepared_full_q_indices = mask_fn_0 = g_prepared_mask_mod_closure_0_cell_contents = None
    out = flex_attention[0]
    lse = flex_attention[1];  flex_attention = None
    _ = lse * 0.6931471805599453;  lse = _ = None
    getitem_11 = out[(slice(None, None, None), slice(None, None, None), slice(None, 12, None), slice(None, None, None))];  out = None
    attn_output = getitem_11.to(dtype = torch.float32);  getitem_11 = None
    transpose_4 = attn_output.transpose(1, 2);  attn_output = None
    attn_output_1 = transpose_4.contiguous();  transpose_4 = None
    attn_output_2 = attn_output_1.reshape(1, -1, 4096);  attn_output_1 = None
    hidden_states_12 = l_z_.to(torch.bfloat16);  l_z_ = None
    att_output = attn_output_2.to(torch.bfloat16)
    getitem_12 = att_output[(slice(None, None, None), slice(0, 8, None))];  att_output = None
    out_emb = torch._C._nn.linear(getitem_12, g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_, None);  getitem_12 = g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = None
    out_emb += hidden_states_12;  out_emb_1 = out_emb;  out_emb = hidden_states_12 = None
    after_first_residual = out_emb_1.clone()
    hidden_states_13 = out_emb_1.to(torch.float32);  out_emb_1 = None
    pow_5 = hidden_states_13.pow(2)
    variance_4 = pow_5.mean(-1, keepdim = True);  pow_5 = None
    add_6 = variance_4 + 1e-06;  variance_4 = None
    rsqrt_4 = torch.rsqrt(add_6);  add_6 = None
    hidden_states_14 = hidden_states_13 * rsqrt_4;  hidden_states_13 = rsqrt_4 = None
    to_19 = hidden_states_14.to(torch.bfloat16);  hidden_states_14 = None
    out_emb_2 = g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ * to_19;  g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = to_19 = None
    linear_7 = torch._C._nn.linear(out_emb_2, g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_, None);  g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = None
    silu = torch.nn.functional.silu(linear_7);  linear_7 = None
    linear_8 = torch._C._nn.linear(out_emb_2, g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_, None);  out_emb_2 = g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = None
    mul_17 = silu * linear_8;  silu = linear_8 = None
    down_proj = torch._C._nn.linear(mul_17, g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_, None);  mul_17 = g_j_modules_qwenvl_modules_model_modules_language_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = None
    down_proj += after_first_residual;  out_emb_3 = down_proj;  down_proj = after_first_residual = None
    hidden_states_15 = l_w_.to(torch.float32);  l_w_ = None
    att_output_1 = attn_output_2.to(torch.float32);  attn_output_2 = None
    getitem_13 = att_output_1[(slice(None, None, None), slice(8, 12, None))];  att_output_1 = None
    out_emb_4 = torch._C._nn.linear(getitem_13, g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_, None);  getitem_13 = g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = None
    out_emb_4 += hidden_states_15;  out_emb_5 = out_emb_4;  out_emb_4 = hidden_states_15 = None
    after_first_residual_1 = out_emb_5.clone()
    hidden_states_16 = out_emb_5.to(torch.float32);  out_emb_5 = None
    pow_6 = hidden_states_16.pow(2)
    variance_5 = pow_6.mean(-1, keepdim = True);  pow_6 = None
    add_7 = variance_5 + 1e-06;  variance_5 = None
    rsqrt_5 = torch.rsqrt(add_7);  add_7 = None
    hidden_states_17 = hidden_states_16 * rsqrt_5;  hidden_states_16 = rsqrt_5 = None
    to_23 = hidden_states_17.to(torch.float32);  hidden_states_17 = None
    out_emb_6 = g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ * to_23;  g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = to_23 = None
    linear_11 = torch._C._nn.linear(out_emb_6, g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_, None);  g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = None
    silu_1 = torch.nn.functional.silu(linear_11);  linear_11 = None
    linear_12 = torch._C._nn.linear(out_emb_6, g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_, None);  out_emb_6 = g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = None
    mul_20 = silu_1 * linear_12;  silu_1 = linear_12 = None
    down_proj_1 = torch._C._nn.linear(mul_20, g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_, None);  mul_20 = g_j_modules_qwen_expert_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = None
    down_proj_1 += after_first_residual_1;  out_emb_7 = down_proj_1;  down_proj_1 = after_first_residual_1 = None
    return (out_emb_3, out_emb_7)
    



def forward(self, L_z_ : torch.Tensor, L_t_ : torch.Tensor, G_a_modules_self_attn_modules_o_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_a_modules_post_attention_layernorm_parameters_weight_ : torch.nn.parameter.Parameter, G_a_modules_mlp_modules_gate_parameters_weight_ : torch.nn.parameter.Parameter, G_a_modules_mlp_buffers_e_score_correction_bias_ : torch.Tensor, G_a_modules_mlp_buffers_avg_topk_sigmoid_score_ : torch.Tensor, G_a_modules_mlp_buffers_tokens_per_expert_ : torch.Tensor, G_a_modules_mlp_modules_experts_parameters_gate_proj_ : torch.nn.parameter.Parameter, G_a_modules_mlp_modules_experts_parameters_up_proj_ : torch.nn.parameter.Parameter, G_a_modules_mlp_modules_experts_parameters_down_proj_ : torch.nn.parameter.Parameter, G_a_modules_mlp_modules_shared_expert_modules_gate_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_a_modules_mlp_modules_shared_expert_modules_up_proj_parameters_weight_ : torch.nn.parameter.Parameter, G_a_modules_mlp_modules_shared_expert_modules_down_proj_parameters_weight_ : torch.nn.parameter.Parameter):
    l_z_ = L_z_
    l_t_ = L_t_
    g_a_modules_self_attn_modules_o_proj_parameters_weight_ = G_a_modules_self_attn_modules_o_proj_parameters_weight_
    g_a_modules_post_attention_layernorm_parameters_weight_ = G_a_modules_post_attention_layernorm_parameters_weight_
    g_a_modules_mlp_modules_gate_parameters_weight_ = G_a_modules_mlp_modules_gate_parameters_weight_
    g_a_modules_mlp_buffers_e_score_correction_bias_ = G_a_modules_mlp_buffers_e_score_correction_bias_
    g_a_modules_mlp_buffers_avg_topk_sigmoid_score_ = G_a_modules_mlp_buffers_avg_topk_sigmoid_score_
    g_a_modules_mlp_buffers_tokens_per_expert_ = G_a_modules_mlp_buffers_tokens_per_expert_
    g_a_modules_mlp_modules_experts_parameters_gate_proj_ = G_a_modules_mlp_modules_experts_parameters_gate_proj_
    g_a_modules_mlp_modules_experts_parameters_up_proj_ = G_a_modules_mlp_modules_experts_parameters_up_proj_
    g_a_modules_mlp_modules_experts_parameters_down_proj_ = G_a_modules_mlp_modules_experts_parameters_down_proj_
    g_a_modules_mlp_modules_shared_expert_modules_gate_proj_parameters_weight_ = G_a_modules_mlp_modules_shared_expert_modules_gate_proj_parameters_weight_
    g_a_modules_mlp_modules_shared_expert_modules_up_proj_parameters_weight_ = G_a_modules_mlp_modules_shared_expert_modules_up_proj_parameters_weight_
    g_a_modules_mlp_modules_shared_expert_modules_down_proj_parameters_weight_ = G_a_modules_mlp_modules_shared_expert_modules_down_proj_parameters_weight_
    hidden_states = l_z_.to(torch.float32);  l_z_ = None
    att_output = l_t_.to(torch.float32);  l_t_ = None
    getitem = att_output[(slice(None, None, None), slice(8, 12, None))];  att_output = None
    out_emb = torch._C._nn.linear(getitem, g_a_modules_self_attn_modules_o_proj_parameters_weight_, None);  getitem = g_a_modules_self_attn_modules_o_proj_parameters_weight_ = None
    out_emb += hidden_states;  out_emb_1 = out_emb;  out_emb = hidden_states = None
    after_first_residual = out_emb_1.clone()
    hidden_states_1 = out_emb_1.to(torch.float32);  out_emb_1 = None
    pow_1 = hidden_states_1.pow(2)
    variance = pow_1.mean(-1, keepdim = True);  pow_1 = None
    add = variance + 1e-06;  variance = None
    rsqrt = torch.rsqrt(add);  add = None
    hidden_states_2 = hidden_states_1 * rsqrt;  hidden_states_1 = rsqrt = None
    to_3 = hidden_states_2.to(torch.float32);  hidden_states_2 = None
    out_emb_2 = g_a_modules_post_attention_layernorm_parameters_weight_ * to_3;  g_a_modules_post_attention_layernorm_parameters_weight_ = to_3 = None
    hidden_flat = out_emb_2.reshape(-1, 768);  out_emb_2 = None
    _enter_autocast = torch.amp.autocast_mode._enter_autocast('cuda', None, False, None)
    float_1 = hidden_flat.float()
    float_2 = g_a_modules_mlp_modules_gate_parameters_weight_.float();  g_a_modules_mlp_modules_gate_parameters_weight_ = None
    router_logits = torch._C._nn.linear(float_1, float_2);  float_1 = float_2 = None
    _exit_autocast = torch.amp.autocast_mode._exit_autocast(_enter_autocast);  _enter_autocast = _exit_autocast = None
    routing_scores = router_logits.sigmoid()
    unsqueeze = g_a_modules_mlp_buffers_e_score_correction_bias_.unsqueeze(0);  g_a_modules_mlp_buffers_e_score_correction_bias_ = None
    scores_for_choice = routing_scores + unsqueeze;  unsqueeze = None
    topk = torch.topk(scores_for_choice, 4, dim = -1);  scores_for_choice = None
    selected_experts = topk[1];  topk = None
    routing_weights = routing_scores.gather(1, selected_experts);  routing_scores = None
    _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None
    detach = routing_weights.detach()
    float_3 = detach.float();  detach = None
    avg_score = float_3.mean();  float_3 = None
    reshape_as = avg_score.reshape_as(g_a_modules_mlp_buffers_avg_topk_sigmoid_score_);  avg_score = None
    to_4 = reshape_as.to(device = device(type='cuda', index=0), dtype = torch.float32);  reshape_as = None
    copy_ = g_a_modules_mlp_buffers_avg_topk_sigmoid_score_.copy_(to_4);  g_a_modules_mlp_buffers_avg_topk_sigmoid_score_ = to_4 = copy_ = None
    detach_1 = selected_experts.detach()
    reshape_1 = detach_1.reshape(-1);  detach_1 = None
    one_hot = torch._C._nn.one_hot(reshape_1, num_classes = 32);  reshape_1 = None
    counts = one_hot.sum(dim = 0);  one_hot = None
    to_5 = counts.to(device = device(type='cuda', index=0), dtype = torch.float32);  counts = None
    add_ = g_a_modules_mlp_buffers_tokens_per_expert_.add_(to_5);  g_a_modules_mlp_buffers_tokens_per_expert_ = to_5 = add_ = None
    _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
    sum_2 = routing_weights.sum(dim = -1, keepdim = True)
    add_2 = sum_2 + 1e-20;  sum_2 = None
    routing_weights_1 = routing_weights / add_2;  routing_weights = add_2 = None
    routing_weights_2 = routing_weights_1 * 4.0;  routing_weights_1 = None
    routing_weights_3 = routing_weights_2.to(torch.float32);  routing_weights_2 = None
    hidden_states_3 = hidden_flat.to(torch.bfloat16)
    routing_weights_4 = routing_weights_3.to(torch.bfloat16);  routing_weights_3 = None
    fc1_1_weight = g_a_modules_mlp_modules_experts_parameters_gate_proj_.to(torch.bfloat16);  g_a_modules_mlp_modules_experts_parameters_gate_proj_ = None
    fc1_2_weight = g_a_modules_mlp_modules_experts_parameters_up_proj_.to(torch.bfloat16);  g_a_modules_mlp_modules_experts_parameters_up_proj_ = None
    fc2_weight = g_a_modules_mlp_modules_experts_parameters_down_proj_.to(torch.bfloat16);  g_a_modules_mlp_modules_experts_parameters_down_proj_ = None
    fwd_body_0 = self.fwd_body_0
    bwd_body_0 = self.bwd_body_0
    final_hidden_states = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, routing_weights_4, selected_experts, hidden_states_3, fc1_1_weight, fc1_2_weight, fc2_weight, args_tensor_mask = [False, True, True, True, True, True, True], non_differentiable_idx = []);  fwd_body_0 = bwd_body_0 = routing_weights_4 = selected_experts = hidden_states_3 = fc1_1_weight = fc1_2_weight = fc2_weight = None
    final_hidden_states_1 = final_hidden_states.to(torch.float32);  final_hidden_states = None
    linear_2 = torch._C._nn.linear(hidden_flat, g_a_modules_mlp_modules_shared_expert_modules_gate_proj_parameters_weight_, None);  g_a_modules_mlp_modules_shared_expert_modules_gate_proj_parameters_weight_ = None
    silu = torch.nn.functional.silu(linear_2);  linear_2 = None
    linear_3 = torch._C._nn.linear(hidden_flat, g_a_modules_mlp_modules_shared_expert_modules_up_proj_parameters_weight_, None);  hidden_flat = g_a_modules_mlp_modules_shared_expert_modules_up_proj_parameters_weight_ = None
    mul_3 = silu * linear_3;  silu = linear_3 = None
    shared_expert_output = torch._C._nn.linear(mul_3, g_a_modules_mlp_modules_shared_expert_modules_down_proj_parameters_weight_, None);  mul_3 = g_a_modules_mlp_modules_shared_expert_modules_down_proj_parameters_weight_ = None
    final_hidden_states_2 = final_hidden_states_1 + shared_expert_output;  final_hidden_states_1 = shared_expert_output = None
    final_hidden_states_3 = final_hidden_states_2.reshape(1, 4, 768);  final_hidden_states_2 = None
    final_hidden_states_3 += after_first_residual;  out_emb_3 = final_hidden_states_3;  final_hidden_states_3 = after_first_residual = None
    return (out_emb_3, router_logits)
    
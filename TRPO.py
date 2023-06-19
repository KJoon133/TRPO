import numpy as np
import os
import gymnasium as gym
import torch
import math
from model import Policy, Value
from running_state import ZFilter
from collections import deque
from tensorboardX import SummaryWriter


gamma = 0.99
hidden = 64
value_lr = 0.0003
batch_size = 64
l2_rate = 0.001
max_kl = 0.01
clip_param = 0.2

def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action


def get_advants(rewards, values):
    rewards = torch.Tensor(rewards)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for i in reversed(range(0, len(rewards))):
        running_returns = rewards[i] + gamma * running_returns 
        running_tderror = rewards[i] + gamma * previous_value  - values.data[i]
        running_advants = running_tderror + gamma * running_advants 

        returns[i] = running_returns
        previous_value = values.data[i]
        advants[i] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)


def surrogate_loss(policy, advants, states, old_policy, actions):
    mu, std, logstd = policy(torch.Tensor(states))
    #print(action)
    new_policy = log_density(torch.Tensor(actions), mu, std, logstd)
    advants = advants.unsqueeze(1)
    surrogate = advants * torch.exp(new_policy - old_policy)
    surrogate = surrogate.mean()
    return surrogate

def train_value(value, states, returns, advants, value_optim):
    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    for epoch in range(5):  
        np.random.shuffle(arr)

        for i in range(n // batch_size):
            batch_index = arr[batch_size * i: batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)  #64bit 정수
            inputs = torch.Tensor(states)[batch_index]
            target1 = returns[batch_index]
            target2 = advants[batch_index]
            target1 = returns.unsqueeze(1)[batch_index]
            target2 = advants.unsqueeze(1)[batch_index]

            values = value(inputs)
            loss = criterion(values, target1 + target2)
            value_optim.zero_grad()
            loss.backward()
            value_optim.step()

def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten

def kl_divergence(new_policy, old_policy, states):

    mu, std, logstd = new_policy(torch.Tensor(states))
    mu_old, std_old, logstd_old = old_policy(torch.Tensor(states))
    #print(mu)
    # mu = mu.detach()
    # std = std.detach()
    # logstd = logstd.detach()

    mu_old = mu_old.detach()
    std_old = std_old.detach()
    logstd_old = logstd_old.detach()
    #print(mu_old)
    kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
         (2.0 * std.pow(2)) - 0.5
    #print(kl)
    return kl.sum(1, keepdim=True)

def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten

def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten

def  fisher_vector_product(policy, states, p):
    p.detach()
    kl = kl_divergence(new_policy=policy, old_policy=policy, states=states)
    kl = kl.mean()
    kl_grad = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
    kl_grad = flat_grad(kl_grad)

    kl_grad_p = (kl_grad * p).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, policy.parameters())
    kl_hessian_p = flat_hessian(kl_hessian_p)

    return kl_hessian_p + 0.1 * p


# openai baseline
def conjugate_gradient(policy, states, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = fisher_vector_product(policy, states, p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length

def train_model(policy, value, memory, value_optim):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    values = value(torch.Tensor(states))
    #policy = torch.tensor(policy)
    #print(policy)

    returns, advants = get_advants(rewards, values)

    train_value(value, states, returns, advants, value_optim)

    mu, std, logstd = policy(torch.Tensor(states))
    old_policy = log_density(torch.Tensor(actions), mu, std, logstd)
    # print(mu)
    loss = surrogate_loss(policy, advants, states, old_policy.detach(), actions)
    loss_grad = torch.autograd.grad(loss, policy.parameters())
    loss_grad = flat_grad(loss_grad)
    step_dir = conjugate_gradient(policy, states, loss_grad.data, nsteps=10)

    params = flat_params(policy)
    shs = 0.5 * (step_dir * fisher_vector_product(policy, states, step_dir)
                 ).sum(0, keepdim=True)
    step_size = 1 / torch.sqrt(shs / max_kl)[0]
    full_step = step_size * step_dir

    old_actor = Policy(policy.num_inputs, policy.num_outputs)
    update_model(old_actor, params)
    expected_improve = (loss_grad * full_step).sum(0, keepdim=True)

    flag = False
    fraction = 1.0
    for i in range(10):
        new_params = params + fraction * full_step
        update_model(policy, new_params)
        new_loss = surrogate_loss(policy, advants, states, old_policy.detach(),
                                  actions)
        loss_improve = new_loss - loss
        expected_improve *= fraction
        #print(policy)
        #print(old_actor)
        kl = kl_divergence(new_policy=policy, old_policy=old_actor, states=states)
        kl = kl.mean()

        print('kl: {:.4f}  loss improve: {:.4f}  expected improve: {:.4f}  '
              'number of line search: {}'
              .format(kl.data.numpy(), loss_improve, expected_improve[0], i))

        if kl < max_kl and (loss_improve / expected_improve) > 0.5:
            flag = True
            break

        fraction *= 0.5

    if not flag:
        params = flat_params(old_actor)
        update_model(policy, params)
        print('policy update does not impove the surrogate')

env = gym.make("Hopper-v4")#, render_mode="human")
observation, info = env.reset(seed=500)

input_num = env.observation_space.shape[0]
action_num = env.action_space.shape[0]

writer = SummaryWriter('logs')

policy = Policy(input_num, action_num)
value = Value(input_num)

value_optim = torch.optim.Adam(value.parameters(), lr = value_lr, weight_decay = l2_rate)

running_state = ZFilter((input_num,), clip=5)

# load_model = None #'ckpt_697.pth.tar'
# if load_model is not None:
#         saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', load_model)
#         ckpt = torch.load(saved_ckpt_path)

#         policy.load_state_dict(ckpt['policy'])
#         value.load_state_dict(ckpt['value'])

#         running_state.rs.n = ckpt['z_filter_n']
#         running_state.rs.mean = ckpt['z_filter_m']
#         running_state.rs.sum_square = ckpt['z_filter_s']

#         print("Loaded OK ex. Zfilter N : ",running_state.rs.n)


episodes = 0
for iter in range(200): #200
    policy.eval(), value.eval()
    memory = deque()


    steps = 0
    scores = []
    while steps < 1000000: #50000
        episodes += 1
        state = env.reset(seed = 500)
        state = running_state(state[0])
        score = 0
        for _ in range(1000):
            
            steps += 1
            mu, std, _ = policy(torch.Tensor(state).unsqueeze(0))
            action = get_action(mu, std)[0]
            # next_state = env.step(action)[0]
            # reward = env.step(action)[1]    
            # done = env.step(action)[2]

            next_state, reward, done, _, _ = env.step(action)
            next_state = running_state(next_state)

            memory.append([state, action, reward])
            #print(action, next_state ,reward)

            score += reward
            state = next_state

            if done:
                break
            
            # if steps > 50000:
            #     break
        scores.append(score)
    score_avg = np.mean(scores)
    print('{} episode score is {:.2f}'.format(episodes, score_avg))
    writer.add_scalar('log/score', float(score_avg), iter)
    policy.train(), value.train()

    # mu1,st1, logstd1 = policy(torch.Tensor(state))
    # print(mu1)

    train_model(policy, value, memory, value_optim)


    score_avg = int(score_avg)

    model_path = os.path.join(os.getcwd(),'save_model')
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')
    print(ckpt_path)
    torch.save({
        'policy': policy.state_dict(),
        'value': value.state_dict(),
        'z_filter_n':running_state.rs.n,
        'z_filter_m': running_state.rs.mean,
        'z_filter_s': running_state.rs.sum_square,
        'score': score_avg
    },ckpt_path)

    # if iter % 100 == 1:
    #     score_avg = int(score_avg)

    #     model_path = os.path.join(os.getcwd(),'save_model')
    #     if not os.path.isdir(model_path):
    #         os.makedirs(model_path)

    #     ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')
    #     print(ckpt_path)
    #     torch.save({
    #         'policy': policy.state_dict(),
    #         'value': value.state_dict(),
    #         'z_filter_n':running_state.rs.n,
    #         'z_filter_m': running_state.rs.mean,
    #         'z_filter_s': running_state.rs.sum_square,
    #         'score': score_avg
    #     },ckpt_path)

env.close() 
def render_condition(model, x_T, sampler, x_start=None, cond=None, h_cond=None):
    if cond is None:
        cond = model.encode(x_start)['cond']
    return sampler.sample(model=model, noise=x_T, model_kwargs={'cond': cond, 'h_cond': h_cond})

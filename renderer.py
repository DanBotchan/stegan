def render_condition(model, x_T, sampler, x_start=None, cond=None, ):
    if cond is None:
        cond = model.encode(x_start)
    return sampler.sample(model=model, noise=x_T, model_kwargs={'cond': cond})

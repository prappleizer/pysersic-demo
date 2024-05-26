import numpy as np
from jax.random import (
    PRNGKey,  # Need to use a seed to start jax's random number generation
)
from pysersic import FitSingle
from pysersic.loss import student_t_loss
from pysersic.priors import autoprior


def run_pysersic(out, band_fit=1):
    try:
        prior = autoprior(
            image=out[band_fit].image,
            profile_type="sersic",
            mask=out[band_fit].mask,
            sky_type="flat",
        )
        print(prior)
        # renormalize psf in case it was cropped
        psf = out[band_fit].psf / np.sum(out[band_fit].psf)

        fitter = FitSingle(
            data=out[band_fit].image,
            rms=out[band_fit].unc,
            mask=out[band_fit].mask,
            psf=psf,
            prior=prior,
            loss_func=student_t_loss,
        )
        fitter.estimate_posterior(PRNGKey(10))
        return fitter

    except Exception as e:
        print(f"{e}")

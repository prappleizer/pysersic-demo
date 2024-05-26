import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import TwoSlopeNorm
from pysersic.rendering import HybridRenderer

from run_pysersic import run_pysersic
from saga_fitting import LegacyCutout, mask_and_crop, parse_coordinate_string

st.set_page_config(layout="wide")


if "cutout" not in st.session_state.keys():
    st.session_state["cutout"] = None
if "crop" not in st.session_state.keys():
    st.session_state["crop"] = None


st.title("Pysersic Legacy Fitter [beta]")

st.write(
    "Fit any* galaxy from the Legacy Surveys using [Pysersic (Pasha & Miller 2023)](https://pysersic.readthedocs.io/en/latest/)."
)
st.write(
    "*This Applet is for demonstration purposes only. For better control of masking, cropping, priors, and rendering, install `pysersic` locally.*"
)
with st.sidebar:
    st.header("Instructions")
    st.write(
        "Begin by entering coordinates and hitting 'retrieve' to pull cutouts of the images, weight maps, and psfs from Legacy. Then select a band and hit 'Fit Galaxy' to run pysersic."
    )
    st.write(
        "The program will attempt to mask, crop, and select priors for the input galaxy (it helps if the galaxy is in the center)."
    )
    st.write(
        "Summary metrics at the top differ from the presented table in that (1) r_eff in pixels is converted to arcsec via the survey pixel scale of 0.262'', (2) ellip is converted to B.A, (3) Flux is converted to mags, (4) PA is converted from radians to degrees."
    )

    st.header("Caveats* / Known Issues")
    st.markdown(
        "- The cutout service pulls from bricks (to obtain weight maps), so galaxies at a brick edge may be cut off/not square (can't be fit in this framework)."
    )
    st.markdown(
        "- On occasion a band has trouble fitting due to something in the weight map or image. Try out other bands if it fails."
    )
    st.markdown(
        "- Note that AB Magnitudes are *not* corrected for galactic extinction or any needed k-correction. They are computed directly assuming a zerpoint of 22.5"
    )
    st.markdown(
        "*Imaging curtesy of the Legacy Survey [(Dey et al. 2019)](https://ui.adsabs.harvard.edu/abs/2019AJ....157..168D/abstract)*"
    )
cols = st.columns([0.25, 0.75])
with cols[0]:
    with st.form(key="coord submit"):
        st.text("Enter coordinates below \nin equitorial coordinates (deg or hmsdms)")
        radec = st.text_input("Coordinates")
        submit = st.form_submit_button("Retrieve Legacy Cutout")
    with st.form(key="band-select"):
        st.text("Choose band to fit.")
        fit_band = st.selectbox("Bands", ["g", "r", "z"])
        fit_submit = st.form_submit_button("Fit Galaxy")

if submit:
    skycoord = parse_coordinate_string(radec)
    cutout = LegacyCutout(
        str_id="User Galaxy", ra=skycoord.ra.value, dec=skycoord.dec.value
    )
    cutout.request_cutout()
    st.write("Cutout Retrieved from Legacy.")
    st.session_state["cutout"] = cutout
    with cols[1]:
        fig, axes = cutout.plot()
        st.pyplot(fig)

if fit_submit:
    if fit_band == "g":
        band_fit = 0
    elif fit_band == "r":
        band_fit = 1
    elif fit_band == "z":
        band_fit = 2
    st.session_state["out"] = mask_and_crop(
        st.session_state.cutout, "r", crop_factor=2.5, overlap_threshold=0.04
    )

    fitter = run_pysersic(st.session_state["out"], band_fit=band_fit)
    summary = fitter.svi_results.summary()
    param_dict = {}
    for a, b in zip(summary.index, summary["mean"]):
        param_dict[a] = b
    im = st.session_state["out"][band_fit].image
    psf = st.session_state["out"][band_fit].psf
    mask = st.session_state["out"][band_fit].mask
    masked_im = np.ma.masked_array(im, mask=mask)
    bf_model = HybridRenderer(im.shape, jnp.array(psf.astype(float))).render_source(
        param_dict, profile_type="sersic"
    )
    masked_model = np.ma.masked_array(np.array(bf_model), mask=mask)
    with cols[1]:
        nc = st.columns(5)
        with nc[0]:
            st.metric(
                "AB Magnitude",
                f"{22.5 - 2.5 * np.log10(param_dict['flux']):.2f}",
            )
        with nc[1]:
            st.metric("Sérsic Index", f"{param_dict['n']:.2f}")
        with nc[2]:
            st.metric(
                "Size (effective radius)",
                f"{param_dict['r_eff'] * 0.262:.2f}''",
            )
        with nc[3]:
            st.metric("Axis Ratio (B/A)", f"{1-param_dict['ellip']:.2f}")
        with nc[4]:
            st.metric(
                "Position Angle (PA)", f"{np.rad2deg(param_dict['theta']):.2f} deg"
            )
        fig, ax = plt.subplots(1, 3, figsize=(20, 6))
        vmin, vmax = np.nanpercentile(im, [5, 98])
        ax[0].imshow(masked_im, origin="lower", vmin=vmin, vmax=vmax)
        ax[1].imshow(masked_model, origin="lower", vmin=vmin, vmax=vmax)
        resid = im - np.array(bf_model)
        masked_residual = np.ma.masked_array(resid, mask=mask)
        resid_scale = np.std(resid)
        ax[2].imshow(
            masked_residual,
            origin="lower",
            cmap="RdBu",
            norm=TwoSlopeNorm(
                vcenter=0, vmin=-2.5 * resid_scale, vmax=2.5 * resid_scale
            ),
        )
        st.pyplot(fig)
        st.table(summary)

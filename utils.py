import functools
from collections import Counter

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sep
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from photutils.aperture import EllipticalAperture


def retry(attempts=3):
    def retry_decorator(func):
        @functools.wraps(func)
        def func_with_retries(*args, **kwargs):
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(
                        f"Attempt {attempt + 1} failed with exception: {e}. Retrying..."
                    )
                    if attempt == attempts - 1:  # If last attempt, raise the exception
                        raise

        return func_with_retries

    return retry_decorator


def process_fits_header(header):
    """Process a FITS header to make it compatible with YAML serialization."""
    processed_header = {}
    for key, value in header.items():
        if isinstance(
            value, fits.header._HeaderCommentaryCards
        ):  # Special handling for commentary cards
            processed_header[key] = str(value)
        elif isinstance(value, (float, int)):  # Directly use floats and ints
            processed_header[key] = value
        else:  # Convert everything else to string
            processed_header[key] = str(value).strip()
    return processed_header


class Band:
    def __init__(self, name, image, wcs, unc, psf, mask=None):
        self.name = name
        self.image = image
        self.wcs = wcs
        self.unc = unc
        self.mask = mask
        self.psf = psf


class LegacyCutout:
    def __init__(self, ra=None, dec=None, str_id=None, im_dir=None):
        self.ra = ra
        self.dec = dec
        self.str_id = str_id
        self.im_dir = im_dir

    def plot(self, figsize=(15, 15), percentiles=[10, 99]):
        fig = plt.figure(figsize=figsize)

        # WCSAxes for the first subplot in each row
        ax = fig.add_subplot(3, 3, 1, projection=self.g.wcs)
        vmin, vmax = np.percentile(self.g.image, percentiles)
        ax.imshow(self.g.image, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.text(
            0.05,
            0.95,
            self.g.name,
            transform=ax.transAxes,
            va="top",
            fontsize=20,
            color="w",
        )
        ax.set_xlabel("Right Ascension")
        ax.set_ylabel("Declination")
        ax2 = fig.add_subplot(
            3,
            3,
            2,
        )
        vmin, vmax = np.percentile(self.g.unc, percentiles)
        ax2.imshow(self.g.unc, origin="lower", vmin=vmin, vmax=vmax, cmap="gray_r")
        ax3 = fig.add_subplot(3, 3, 3)
        vmin, vmax = np.percentile(self.g.psf, percentiles)
        ax3.imshow(self.g.psf, origin="lower", vmin=vmin, vmax=vmax, cmap="magma")

        ax4 = fig.add_subplot(3, 3, 4, projection=self.r.wcs)
        vmin, vmax = np.percentile(self.r.image, percentiles)
        ax4.imshow(self.r.image, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax4.text(
            0.05,
            0.95,
            self.r.name,
            transform=ax4.transAxes,
            va="top",
            fontsize=20,
            color="w",
        )
        ax5 = fig.add_subplot(3, 3, 5)
        vmin, vmax = np.percentile(self.r.unc, percentiles)
        ax5.imshow(self.r.unc, origin="lower", vmin=vmin, vmax=vmax, cmap="gray_r")

        ax6 = fig.add_subplot(3, 3, 6)
        vmin, vmax = np.percentile(self.r.psf, percentiles)
        ax6.imshow(self.r.psf, origin="lower", vmin=vmin, vmax=vmax, cmap="magma")

        ax7 = fig.add_subplot(3, 3, 7, projection=self.z.wcs)
        vmin, vmax = np.percentile(self.z.image, percentiles)
        ax7.imshow(self.z.image, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax7.text(
            0.05,
            0.95,
            self.z.name,
            transform=ax7.transAxes,
            va="top",
            fontsize=20,
            color="w",
        )
        ax8 = fig.add_subplot(3, 3, 8)
        vmin, vmax = np.percentile(self.z.unc, percentiles)
        ax8.imshow(self.z.unc, origin="lower", vmin=vmin, vmax=vmax, cmap="gray_r")

        ax9 = fig.add_subplot(3, 3, 9)
        vmin, vmax = np.percentile(self.z.psf, percentiles)
        ax9.imshow(self.z.psf, origin="lower", vmin=vmin, vmax=vmax, cmap="magma")

        axes = [ax, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
        return fig, axes

    @retry(attempts=3)
    def request_cutout(
        self,
        size: int = 300,
        write: bool = False,
        out_filename: str = None,
        verbose: bool = False,
    ):
        ra = self.ra
        dec = self.dec
        ra_deg = ra
        dec_deg = dec
        uri = f"https://www.legacysurvey.org/viewer-dev/cutout.fits?ra={ra_deg}&dec={dec_deg}&layer=ls-dr9&size={size}&invvar"
        print(uri)
        with fits.open(uri, lazy_load_hdus=False, cache=False, timeout=30) as hdu:
            print(hdu.info())
            g = hdu[0].data[0]
            g_header = hdu[0].header
            g_invar = hdu[1].data[0]
            r = hdu[0].data[1]
            r_header = hdu[0].header
            r_invar = hdu[1].data[1]
            z = hdu[0].data[2]
            z_header = hdu[0].header
            z_invar = hdu[1].data[2]
        try:
            if dec_deg > 32.375:
                uri_psf = f"https://www.legacysurvey.org/viewer-dev/coadd-psf/?ra={ra_deg}&dec={dec_deg}&layer=ls-dr9-north"
                try:
                    with fits.open(uri_psf, timeout=30, lazy_load_hdus=False) as hdu:
                        psf_g = hdu[0].data
                        psf_r = hdu[1].data
                        psf_z = hdu[2].data
                except:
                    uri_psf = uri_psf.replace("north", "south")
                    with fits.open(uri_psf, timeout=30, lazy_load_hdus=False) as hdu:
                        psf_g = hdu[0].data
                        psf_r = hdu[1].data
                        psf_z = hdu[2].data
            else:
                try:
                    uri_psf = f"https://www.legacysurvey.org/viewer-dev/coadd-psf/?ra={ra_deg}&dec={dec_deg}&layer=ls-dr9-south"
                    with fits.open(uri_psf, timeout=30, lazy_load_hdus=False) as hdu:
                        psf_g = hdu[0].data
                        psf_r = hdu[1].data
                        psf_z = hdu[2].data
                except:
                    uri_psf = uri_psf.replace("south", "north")
                    with fits.open(uri_psf, timeout=30, lazy_load_hdus=False) as hdu:
                        psf_g = hdu[0].data
                        psf_r = hdu[1].data
                        psf_z = hdu[2].data
        except:
            print("Could Not Retrieve PSF.")

        g_unc = (g_invar**-1) ** 0.5
        self.g = Band(
            name="g",
            image=g,
            wcs=WCS(g_header.dropaxis(2)),
            unc=g_unc,
            mask=np.isinf(g_unc).astype(int),
            psf=psf_g,
        )
        r_unc = (r_invar**-1) ** 0.5
        self.r = Band(
            name="r",
            image=r,
            wcs=WCS(r_header.dropaxis(2)),
            unc=r_unc,
            mask=np.isinf(r_unc).astype(int),
            psf=psf_r,
        )
        z_unc = (z_invar**-1) ** 0.5
        self.z = Band(
            name="z",
            image=z,
            wcs=WCS(z_header.dropaxis(2)),
            unc=z_unc,
            mask=np.isinf(z_unc).astype(int),
            psf=psf_z,
        )
        return


def crop_center(array, crop_height, crop_width):
    """
    Crop a 2D numpy array to the specified width and height, centered on the array's center.

    Parameters:
    array (np.ndarray): The input 2D array to crop.
    crop_height (int): The desired height of the cropped array.
    crop_width (int): The desired width of the cropped array.

    Returns:
    np.ndarray: The cropped 2D array.
    """
    # Get the dimensions of the original array
    original_height, original_width = array.shape

    # Calculate the center of the original array
    center_y, center_x = original_height // 2, original_width // 2

    # Calculate the starting and ending indices for the crop
    start_x = max(center_x - crop_width // 2, 0)
    end_x = min(center_x + crop_width // 2, original_width)
    start_y = max(center_y - crop_height // 2, 0)
    end_y = min(center_y + crop_height // 2, original_height)

    # Adjust indices if the crop size is odd
    if crop_width % 2 != 0:
        end_x += 1
    if crop_height % 2 != 0:
        end_y += 1

    # Perform the crop
    cropped_array = array[start_y:end_y, start_x:end_x]

    return cropped_array


def mask_and_crop(
    cutout,
    band_name,
    mask=None,
    overlap_threshold=0.01,
    danger_threshold=0.4,
    bright_mask_percentile=80,
    crop=True,
    crop_factor=1.5,
):
    image = getattr(cutout, band_name).image
    wcs = getattr(cutout, band_name).wcs
    psf = getattr(cutout, band_name).psf
    mask = getattr(cutout, band_name).mask.astype(float)
    band_obj = getattr(cutout, band_name)
    original_shape = image.shape
    ra = cutout.ra
    dec = cutout.dec
    try:
        bkg = sep.Background(image, mask=None, bw=64, bh=64, fw=3, fh=3)
    except:
        image = image.byteswap().newbyteorder()
        bkg = sep.Background(image, mask=None, bw=64, bh=64, fw=3, fh=3)

    data_sub = image - bkg
    objects = sep.extract(data_sub, 1.5, err=bkg.globalrms)
    df = pd.DataFrame(objects)

    gal_pos = SkyCoord(ra=ra, dec=dec, unit="deg")
    pos_need = np.array(band_obj.wcs.world_to_pixel(gal_pos)).flatten()

    d2 = (pos_need[0] - objects["x"]) ** 2 + (pos_need[1] - objects["y"]) ** 2
    obj_ind = np.argmin(d2)
    obj_params = df.iloc[obj_ind]

    primary_aperture = EllipticalAperture(
        positions=[[df.loc[obj_ind, "x"], df.loc[obj_ind, "y"]]],
        a=3.5 * df.loc[obj_ind, "a"],
        b=3.5 * df.loc[obj_ind, "b"],
        theta=df.loc[obj_ind, "theta"],
    )

    primary_mask = np.zeros(image.shape, dtype=float)
    primary_mask_image = primary_aperture.to_mask()[0].to_image(primary_mask.shape)
    primary_mask += primary_mask_image
    primary_area = np.sum(primary_mask)

    df = df.drop(obj_ind, axis=0)

    for i in df.index:
        try:
            ap = EllipticalAperture(
                positions=[[df.loc[i, "x"], df.loc[i, "y"]]],
                a=3.5 * df.loc[i, "a"],
                b=3.5 * df.loc[i, "b"],
                theta=df.loc[i, "theta"],
            )

            temp_mask = np.zeros(image.shape, dtype=float)
            temp_mask_image = ap.to_mask()[0].to_image(temp_mask.shape)
            temp_mask += temp_mask_image

            overlap_area = np.sum((primary_mask > 0) & (temp_mask > 0))

            if overlap_area / primary_area <= overlap_threshold:
                bbox = ap.to_mask()[0].bbox
                mask[
                    bbox.iymin : bbox.iymax,
                    bbox.ixmin : bbox.ixmax,
                ] += temp_mask[bbox.iymin : bbox.iymax, bbox.ixmin : bbox.ixmax]
            elif overlap_area / primary_area >= danger_threshold:
                print(
                    f"DANGER : {cutout.str_id} : PROPOSED A MASK THAT COVERS LARGE FRAC OF THE GALAXY"
                )
            elif overlap_area / primary_area > overlap_threshold:
                print(
                    f"WARNING : {cutout.str_id} : SEP wants to add a mask to the galaxy > overlap thresh [{overlap_area / primary_area:.2f}](but less than danger thresh). Ignoring that mask for now."
                )

        except Exception as e:
            print(f"Skipping object due to error: {e}")
            continue

    if mask[int(pos_need[1]), int(pos_need[0])] > 0:
        print("A mask has covered the center of the target galaxy!")
    primary_pixels = image[primary_mask > 0]
    brightness_threshold = np.percentile(primary_pixels, bright_mask_percentile)
    bright_pixels_mask = (image > brightness_threshold) & (primary_mask == 0)
    mask[bright_pixels_mask] = 1
    # Crop the image and mask if crop is True
    if crop:
        a_extent = 3.5 * obj_params["a"]  # Use saved parameters
        crop_size = int(crop_factor * a_extent)
        x_center, y_center = int(obj_params["x"]), int(obj_params["y"])

        position = (y_center, x_center)
        size = (crop_size, crop_size)
        newbands = []
        for i in [cutout.g, cutout.r, cutout.z]:
            newcutout = Cutout2D(data=i.image, position=position, size=size, wcs=wcs)
            image_cropped = newcutout.data
            new_wcs = newcutout.wcs
            mask_cropped = Cutout2D(
                data=mask, position=position, size=size, wcs=wcs
            ).data
            unc_cropped = Cutout2D(
                data=i.unc, position=position, size=size, wcs=wcs
            ).data
            if i.psf.shape[0] > image_cropped.shape[0]:
                print(i.psf.shape)
                print(image_cropped.shape)
                print("WARNING : PSF IM > CUTOUT : CROPPING.")
                psf = crop_center(
                    i.psf, image_cropped.shape[0] - 3, image_cropped.shape[1] - 3
                )
            else:
                psf = i.psf
            newband = Band(
                f"{i.name}_cropped",
                image=image_cropped,
                wcs=new_wcs,
                unc=unc_cropped,
                psf=psf,
                mask=mask_cropped,
            )
            newbands.append(newband)
    return newbands


def parse_coordinate_string(input_string):
    if "," in input_string:
        spl = input_string.split(",")
        if (" " in spl[0].strip()) or (":" in spl[0].strip()):
            reconstituted = spl[0].strip() + " " + spl[1].strip()
            coordinate = SkyCoord(reconstituted, unit=(u.hourangle, u.deg))
        else:
            ra = float(spl[0].strip())
            dec = float(spl[1].strip())
            coordinate = SkyCoord(ra=ra, dec=dec, unit="deg")
    elif ":" in input_string:
        coordinate = SkyCoord(input_string, unit=(u.hourangle, u.deg))
    else:
        nspaces = Counter(input_string)[" "]
        if nspaces > 1:
            coordinate = SkyCoord(input_string, unit=(u.hourangle, u.deg))
        elif nspaces == 1:
            coordinate = SkyCoord(input_string, unit="deg")

    return coordinate

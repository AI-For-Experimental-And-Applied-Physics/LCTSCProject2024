from pathlib import Path
from .dicom_utils import load_ct_series
from .rtstruct_utils import load_rtstruct
from .config import OUTPUT_DIR
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def preprocess_case(case_id: str, ct_path: str, rtstruct_path: str, plot=False):
    ct_dir = Path(ct_path)
    rt_file = Path(rtstruct_path)

    if not ct_dir.exists() or not rt_file.exists():
        print(f"[!] Skipping {case_id}: CT or RTSTRUCT path missing.")
        return

    try:
        image, pixel_dim = load_ct_series(ct_dir)
        rtstruct = load_rtstruct(rt_file, ct_dir)
        roi_names = rtstruct.get_roi_names()

        mask_dict = {}
        for roi in roi_names:
            try:
                mask = rtstruct.get_roi_mask_by_name(roi).astype(np.uint8)
                mask_dict[roi] = mask
            except Exception as e:
                print(f"[!] Could not extract ROI '{roi}' for {case_id}: {e}")

        if mask_dict:
            np.savez_compressed(
                OUTPUT_DIR / f"{case_id}.npz",
                image=image,
                **mask_dict
            )
            print(f"[✓] Saved {case_id} with {len(mask_dict)} ROIs.")
        else:
            print(f"[!] No valid ROIs found for {case_id}.")
        if plot:
            plot_roi_overlays(image, list(mask_dict.values()), list(mask_dict.keys()),pixel_dim)
    except Exception as e:
        print(f"[!] Error processing {case_id}: {e}")

def plot_roi_overlays(image, masks, roi_names, pixel_dim=None):
    print(pixel_dim)
    # Define colors for ROIs
    colors = plt.colormaps.get_cmap('tab10')
    alpha = 0.25  # Transparency for the overlays

    # Create a combined mask for each ROI
    combined_masks = np.zeros_like(image, dtype=np.float32)
    for i, mask in enumerate(masks):
        combined_masks += mask * (i + 1)

    # Create a custom colormap for the ROIs
    cmap = ListedColormap([colors(i) for i in range(len(roi_names)+1)])

    # Plot on all three planes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    planes = ['Axial', 'Sagittal', 'Coronal']

    # Axial plane
    mid_axial = image.shape[0] // 2
    axes[0].imshow(image[mid_axial], cmap='gray', aspect=pixel_dim[1] / pixel_dim[2])   
    axes[0].imshow(combined_masks[mid_axial], cmap=cmap, alpha=alpha, aspect=pixel_dim[1] / pixel_dim[2],vmin=0,vmax=len(roi_names)+1)
    axes[0].set_title(f"{planes[0]} Plane")
    axes[0].axis('off')

    # Sagittal plane
    mid_sagittal = image.shape[2] // 2
    axes[1].imshow(image[:, :, mid_sagittal], cmap='gray', aspect=pixel_dim[0] / pixel_dim[1])
    axes[1].imshow(combined_masks[:, :, mid_sagittal], cmap=cmap, alpha=alpha, aspect=pixel_dim[0] / pixel_dim[1],vmin=0,vmax=len(roi_names)+1)
    axes[1].set_title(f"{planes[1]} Plane")
    axes[1].axis('off')

    # Coronal plane
    mid_coronal = image.shape[1] // 2
    axes[2].imshow(image[:, mid_coronal, :], cmap='gray', aspect=pixel_dim[0] / pixel_dim[2])
    axes[2].imshow(combined_masks[:, mid_coronal, :], cmap=cmap, alpha=alpha, aspect=pixel_dim[0] / pixel_dim[2],vmin=0,vmax=len(roi_names)+1)
    axes[2].set_title(f"{planes[2]} Plane")
    axes[2].axis('off')

    # Add a legend for ROI names
    handles = [plt.Line2D([0], [0], color=colors(i+1), lw=4) for i in range(len(roi_names))]
    fig.legend(handles, roi_names, loc='upper center', ncol=len(roi_names), bbox_to_anchor=(0.5, 0.95))

    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dcm
from rt_utils import RTStructBuilder
import os
import SimpleITK as sitk
import pandas as pd
import shutil
import re
import time
import matplotlib.patches as patches

class Plan():
    def __init__(self, parent_folder):
        """
        Class for CT or CBCT plan. This class will store the info for the different items needed to compare two dosemaps

        Args:
            parent_folder (string): Folder containing the CT slices, RTStruct, RTDose and RTPlan files
        """

        self.folder_name =  parent_folder

        # Create path to store image slices in
        self.image_path = os.path.join(parent_folder,'Slices')
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)
        
        # Define the path for RTDose, RTStruct and Image slices
        files = os.listdir(parent_folder)
        for file in files:
            if "RS" in file:
                self.rtStruct_path = os.path.join(parent_folder,file)
            elif "RD" in file:
                self.rtDose_path = os.path.join(parent_folder,file)
            elif "CT" in file:
                shutil.move(os.path.join(parent_folder,file),self.image_path)
            elif "RP" in file:
                self.rtPlan_path = os.path.join(parent_folder,file)

        # Define the Dose Image and RTStruct objects
        self.rtDose_sitk =sitk.ReadImage(self.rtDose_path) 
        self.rtStruct = RTStructBuilder.create_from(self.image_path,self.rtStruct_path)
        self.roi_names = self.rtStruct.get_roi_names()

        # Get dose map information
        self.spacing = self.rtDose_sitk.GetSpacing()
        self.origin = self.rtDose_sitk.GetOrigin()
        self.size = self.rtDose_sitk.GetSize()
        self.direction = self.rtDose_sitk.GetDirection()
        self.doseGridScaling = float(self.rtDose_sitk.GetMetaData('3004|000e'))

        self.rtDose_arr = sitk.GetArrayFromImage(self.rtDose_sitk) * self.doseGridScaling

        # Define RTPlan object and extract isocenter location
        self.rtPlan = dcm.dcmread(self.rtPlan_path)
        self.isocenter = self.rtPlan['BeamSequence'][0]['ControlPointSequence'][0]['IsocenterPosition'].value

        # Read CT slices to get image informaiton
        image_slices = os.listdir(self.image_path)
        file_name_ct = os.path.join(self.image_path, image_slices[0])

        file_reader_ct=sitk.ImageFileReader()
        file_reader_ct.SetFileName(file_name_ct)
        file_reader_ct.ReadImageInformation()

        series_ID_ct=file_reader_ct.GetMetaData('0020|000e')
        sorted_file_names_ct=sitk.ImageSeriesReader.GetGDCMSeriesFileNames(self.image_path,series_ID_ct) 

        self.image_sitk = sitk.ReadImage(sorted_file_names_ct)


def iso_translation(iso_moving_mm, iso_fixed_mm):
    """
    Calclate the translation needed to align the two dose maps based on 
    the difference between the isocenter positions of the two plans.

    Args:
        iso_moving_mm (list): list of floats showing the isocenter position of the moving dosemap (ex: CBCT dosemap)
        iso_fixed_mm (list): list of floats showing the isocenter position of the reference dosemap (ex: CT dosemap)

    Returns:
        TranslationTransform: TranslationTransform that aligns the moving image
    (CBCT dose) onto the fixed image (CT dose) so their isocenters coincide
    """
    delta = np.array(iso_moving_mm, float) - np.array(iso_fixed_mm, float)
    return sitk.TranslationTransform(3, tuple(delta.tolist()))

def resampleDose_to_refGrid(mod_plan, ref_plan, transform, atol=1e-8):
    """   
    Resample mod_plan (moving) dose to ref_plan (fixed) dose grid.
    Linear for dose, but multiplied by a resampled NN mask so
    no new non-zero slices appear.

    Args:
        mod_plan (Plan): moving dose map object
        ref_plan (Plan): fixed dose map object
        transform (TranslationTransform): the transformation needed to align the mod_plan to the ref_plan
        atol (double, optional): Threshold under which a slice is considered to be filled by 0 values. Defaults to 1e-8.

    Returns:
        SimpleITK Image: resampled dose map 
    """

    # --- build a binary support mask in CBCT space  ---
    support_mask = sitk.Cast(mod_plan.rtDose_sitk > atol, sitk.sitkUInt8)

    # --- resample dose with LINEAR ---
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_plan.rtDose_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(transform)
    out_lin = resampler.Execute(mod_plan.rtDose_sitk)   

    # --- resample mask with NEAREST NEIGHBOR (same settings) ---
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    out_mask = resampler.Execute(support_mask)          # 0/1 in ref grid

    # --- apply mask to kill interpolation bleed at edges/slices ---
    out = sitk.Cast(out_lin, sitk.sitkFloat32) * sitk.Cast(out_mask, sitk.sitkFloat32)
    out.CopyInformation(ref_plan.rtDose_sitk)

    return out

def subtract_dosemaps(eval_dose, ref_dose):
    """Subtract the eval_dose map from the ref_dose map

    Args:
        eval_dose (numpy array): evaluated dose map (ex: CBCT dose map)
        ref_dose (numpy array): reference dose map (ex: CT dose map)

    Returns:
        numpy array: the resulting difference dose map
    """
    # Create a boolean mask of slices that are all zero in CBCT
    zero_slices_mask = np.all(eval_dose == 0, axis=(1, 2))

    # Start with subtraction everywhere
    dose_diff_arr = eval_dose - ref_dose

    # Force zero for slices where CBCT is entirely zero
    dose_diff_arr[zero_slices_mask, :, :] = 0
    
    return dose_diff_arr

def resampleMask_to_refGrid(mask_array ,ref_plan, roi_name, output_path):
    """
    Resample a binary mask image to fit the size of the RT Dosemap. The binary mask originally has the same size as the ct image

    Args:
        mask_array (numpy array): binary numpy array of the mask image
        ref_plan (Plan): Plan object whose rtdose map will be used as the reference grid for the resampling

    Returns:
        numpy array: resampled binary mask array
    """

    # Transpose mask array to fit the order of Sitk images
    mask_t = np.transpose(mask_array,(2,0,1)).astype(np.uint8)
    mask_sitk = sitk.GetImageFromArray(mask_t)
    mask_sitk.CopyInformation(ref_plan.image_sitk) # Copy the Dicom info (spacing, origin, etc...) of the CT Image for the binary mask

    # Resample the binary mask to the size of the CT RT Dose map
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_plan.rtDose_sitk)  
    resampler.SetInterpolator(sitk.sitkNearestNeighbor) # Use NN interpolation to keep binary aspect
    resampler.SetDefaultPixelValue(0)     
    resampler.SetTransform(sitk.Transform(3,sitk.sitkIdentity)) # CT image and Dose map are always aligned

    resampled_mask_sitk = resampler.Execute(mask_sitk)

    # Export the binary mask to output folder
    sitk.WriteImage(resampled_mask_sitk, os.path.join(output_path,f"Mask_{roi_name}.dcm"))

    resampled_mask_arr = sitk.GetArrayFromImage(resampled_mask_sitk).astype(bool)

    return resampled_mask_arr

def calculate_roi_statistics (roi_name, dose_diff_arr, output_path):
    """
    Calculate different statistics for a given ROI within the dose difference array.
    Provides the absolute mean, median and maximum error in a given ROI.

    Args:
        roi_name (string): name of the ROI that will be used as a mask
        dose_diff_arr (numpy array): array representing the difference between the CBCT dose map and CT dose map (subtraction)
    """

    # Create binary mask of the given ROI
    mask_arr = ctPlan.rtStruct.get_roi_mask_by_name(roi_name)

    # Resample the ROI to have the same dimensions as the dose_diff_arr = dimensions of the CT Dosemap
    resampled_mask_arr = resampleMask_to_refGrid(mask_arr,ctPlan, roi_name, output_path)

    # Filter the the dose_diff_arr to only keep the values that are within the binary mask
    dose_diff_arr_filtered = dose_diff_arr[resampled_mask_arr]    

    #Drop 0 values corresponding to slices where the diff dose map is 0 because these slices are out of the cbct dose map region
    dose_diff_arr_filtered = dose_diff_arr_filtered[dose_diff_arr_filtered != 0]

    # Apply absolute value to have all the dose differences in positive
    abs_dose_diff_arr = np.abs(dose_diff_arr_filtered)

    roi_stats={
    "ROI": roi_name,
    "Mean Absolute Error (Gy)": np.mean(abs_dose_diff_arr),
    "Median Absolute Error (Gy)": np.median(abs_dose_diff_arr),
    "Max Absolute Error (Gy)": np.max(abs_dose_diff_arr)
    }

    return roi_stats

def _sanitize_sheet_name(name: str) -> str:
    """
    Remove all unwanted characters for a given sheet name

    Args:
        name (str): sheet name

    Returns:
        str: sanitized sheet name
    """
    # Banned characters: : \ / ? * [ ]
    clean = re.sub(r'[:\\/?*\[\]]', '_', str(name))
    clean = clean.strip()
    if not clean:
        clean = "Sheet"
    return clean[:31]  # Limite Excel 

def _unique_sheet_name(base: str, existing: set) -> str:
    if base not in existing:
        existing.add(base)
        return base
    i = 2
    while True:
        cand = _sanitize_sheet_name(f"{base}_{i}")[:31]
        if cand not in existing:
            existing.add(cand)
            return cand
        i += 1

def plot_max_dose_diff (dose_diff_arr, ct_dose_arr, cbct_dose_arr, output_path, localisation_name, protocol_name):
    """
    Plot and save the slice where the dose difference is at it's maximum
    Args:
        dose_diff_arr (numpy array): Dose difference array
        ct_dose_arr (numpy array): CT dose array
        cbct_dose_arr (numpy array): Resampled CBCT dose array
        output_path (string): Path to output folder
    """

    # Localise max dose difference
    dose_diff_abs = np.abs(dose_diff_arr)
    flat_index_maxdiff = np.argmax(dose_diff_abs)
    coord_max = np.unravel_index(flat_index_maxdiff, dose_diff_arr.shape)
    max_diff_value = dose_diff_abs[coord_max]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"{localisation_name} - {protocol_name} — Max Absolute value dose difference slice", fontsize=14, fontweight='bold')

    # Plot CT dose map
    im0 = axes[0].imshow(ct_dose_arr[coord_max[0], :, :], cmap="jet")
    axes[0].set_title("CT Dose Map (Gy)")
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # CBCT Dose map
    im1 = axes[1].imshow(cbct_dose_arr[coord_max[0], :, :], cmap="jet")
    axes[1].set_title("Resampled CBCT Dose Map (Gy)")
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Dose difference map
    im2 = axes[2].imshow(dose_diff_abs[coord_max[0], :, :], cmap="jet")
    axes[2].set_title(f"Absolute value Dose Difference\n(Max = {max_diff_value:.2f} Gy)\n Circled in yellow")
    axes[2].axis('off')

    # Highlight max difference location
    y, x = coord_max[1], coord_max[2]
    circle = patches.Circle(
        (x, y),                   # (x, y) center
        radius=3.5,                # adjust for desired size
        linewidth=0.5,
        edgecolor='yellow',
        facecolor='none',         # no fill
        alpha=1               # transparency for the edge
    )
    axes[2].add_patch(circle)

    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = os.path.join(output_path, f'{localisation_name}_{protocol_name}_max_dose_difference_plot.pdf')
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    # plt.show()

############################################################## M A I N ###############################################

root_folder = ".\\Input"

 # Step 1: Go through folders and create CT and CBCT Plan objects (folder architecture: Root -> Localisations -> CBCT Protocols -> CT & CBCT folders)
try:
    for localisation_folder in os.listdir(root_folder):
        loc_folder_path = os.path.join(root_folder, localisation_folder)

        # Ensure we only iterate through directories
        if not os.path.isdir(loc_folder_path):
            continue
        
        #Initiale output excel file containing the roi stats
        excel_filename = f"stats_{_sanitize_sheet_name(localisation_folder)}.xlsx"
        excel_path = os.path.join(loc_folder_path, excel_filename)
        sheet_names_seen = set()
        writer = pd.ExcelWriter(excel_path, engine="openpyxl")


        try:
            for protocol_folder in os.listdir(loc_folder_path):
                protocol_path = os.path.join(loc_folder_path, protocol_folder)
                
                print("******************************************************")
                print(f"1. Analysing {protocol_path}")

                # Ensure it's a directory
                if not os.path.isdir(protocol_path):
                    continue
                try:
                    for fff in os.listdir(protocol_path):
                        if  fff == 'CT':
                            ctPlan = Plan(os.path.join(protocol_path,fff))
                            print(f'\t CT Plan object created from {os.path.join(protocol_path,fff)}')
                        elif  fff == 'CBCT':
                            cbctPlan = Plan(os.path.join(protocol_path,fff))
                            print(f'\t CBCT Plan object created from {os.path.join(protocol_path,fff)}')
                except Exception as e:
                    print(f"Error creating Plan for {protocol_path}: {e}")
                    continue  # Skip this protocol folder

                # Create a path to save output files
                try:
                    output_path = os.path.join(protocol_path, 'output')
                    os.makedirs(output_path, exist_ok=True)
                except Exception as e:
                    print(f"Error creating output folder {output_path}: {e}")

                # Step 2: Register and resample CBCT Dosemap to CT Dosemap
                print("2. CBCT to CT Registration")
                print("\t Registering and resampling CBCT dose map to CT Dosemap")
                tx_iso = iso_translation(cbctPlan.isocenter, ctPlan.isocenter)
                print(f"\t\t Registration shift (X,Y,Z) in mm: {tx_iso.GetOffset()}")
                cbct_dose_resampled = resampleDose_to_refGrid(cbctPlan, ctPlan, tx_iso)
                resampled_cbct_dose_arr = sitk.GetArrayFromImage(cbct_dose_resampled) * cbctPlan.doseGridScaling

                # Save the resampled cbct dose map as an image
                cbct_dose_resampled_scaled = sitk.GetImageFromArray(resampled_cbct_dose_arr) #Regenerate image to include doseGridScaling
                sitk.WriteImage(cbct_dose_resampled_scaled, os.path.join(output_path,f"{localisation_folder}_{protocol_folder}_cbct_dose_resampled.nii.gz"))
                print(f"\t Resampled CBCT dose map saved to: {os.path.join(output_path,f"{localisation_folder}_{protocol_folder}_cbct_dose_resampled.nii.gz")}")

                # Step 3: Subtract the resampled cbct dose map and ct dose map
                print('3. Performing CBCT - CT subtraction')
                resampled_cbct_dose_arr = sitk.GetArrayFromImage(cbct_dose_resampled) * cbctPlan.doseGridScaling
                ct_dose_arr = ctPlan.rtDose_arr

                dose_diff_arr = subtract_dosemaps(resampled_cbct_dose_arr,ct_dose_arr)

                # Save the difference dose map as an image
                dose_diff_image = sitk.GetImageFromArray(dose_diff_arr)
                dose_diff_image.CopyInformation(ctPlan.rtDose_sitk)
                sitk.WriteImage(dose_diff_image, os.path.join(output_path,f"{localisation_folder}_{protocol_folder}_CBCT_minus_CT.nii.gz"))

                #Save a figure of the max dose difference
                plot_max_dose_diff(dose_diff_arr,ct_dose_arr,resampled_cbct_dose_arr,output_path, localisation_folder, protocol_folder)

                # Step 4: Calculate the error in each structure
                stat_df = pd.DataFrame(columns=[
                            "ROI",
                            "Mean Absolute Error (Gy)",
                            "Median Absolute Error (Gy)",
                            "Max Absolute Error (Gy)"
                        ]) # output dataframe

                for structure in (s for s in ctPlan.roi_names if "Couch" not in s and "ORFIT" not in s) :
                    try:
                        roi_stats= calculate_roi_statistics(structure,dose_diff_arr, output_path)
                        stat_df.loc[len(stat_df)] = roi_stats #Add the structure stats to output datafarame
                    except AttributeError as e:
                        if "'Dataset' object has no attribute 'ContourSequence'" in str(e):
                            print(f"Skipping {structure}: no contours found.")
                            continue  # Skip to next ROI
                        else:
                             raise  # Re-raise if it's another kind of AttributeError
                        
                # Write to output excel file
                sheet_base = _sanitize_sheet_name(protocol_folder)
                sheet_name = _unique_sheet_name(sheet_base, sheet_names_seen)
                stat_df = stat_df.sort_values(by="ROI", kind="mergesort")
                stat_df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"\t Wrote stats to {excel_path} -> sheet '{sheet_name}'")

                time.sleep(3)


        except Exception as e:
            print(f"Error listing protocol folders in {loc_folder_path}: {e}")

        finally:
            try:
                writer.close()
                print(f"Saved Excel file for localisation '{localisation_folder}': {excel_path}")
            except Exception as e:
                print(f"Error saving Excel file {excel_path}: {e}")

except Exception as e:
    print(f"Error listing folders in {root_folder}: {e}")


# slice_idx = 189

# fig, ax = plt.subplots(figsize=(6,6))

# # Base image: dose difference with jet colormap
# im = ax.imshow(dose_diff_arr[slice_idx, :, :],
#                cmap='jet')  # origin='lower' if you want patient-friendly orientation

# # Overlay: binary mask as translucent
# # We'll map True -> 1, False -> 0, and use alpha for transparency
# # mask_overlay = np.ma.masked_where(~resampled_mask_array[slice_idx, :, :],
# #                                   resampled_mask_array[slice_idx, :, :])
# ax.imshow(resampled_mask_arr[slice_idx, :, :],
#           cmap='Reds',    # or another color (e.g. 'Reds')
#           alpha=0.3)

# # Add colorbar for the dose
# cbar = plt.colorbar(im, ax=ax)
# cbar.set_label('Dose difference (Gy)')

# ax.set_title(f"Slice {slice_idx} - Dose diff with mask overlay")
# plt.tight_layout()
# plt.show()




# # Localise max dose difference
# dose_diff_abs = np.abs(dose_diff_arr)
# flat_index_maxdiff = np.argmax(dose_diff_abs)
# coord_max = np.unravel_index(flat_index_maxdiff, dose_diff.shape)


# plt.figure()
# plt.subplot(2,2,1)
# plt.imshow(ct_dose_arr[coord_max[0],:,:], cmap="jet")
# plt.title("CT Dose map")

# plt.subplot(2,2,2)
# plt.imshow(cbct_dose_arr[coord_max[0],:,:], cmap="jet")
# plt.title("Resampled CBCT Dose map")

# plt.subplot(2,2,3)
# plt.imshow(dose_diff_abs[coord_max[0],:,:], cmap = "jet")
# plt.colorbar()
# plt.title(f"Max dose difference: {np.max(dose_diff_abs)}")

# plt.tight_layout()
# plt.show()

pass

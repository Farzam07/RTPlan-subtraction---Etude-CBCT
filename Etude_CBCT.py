import numpy as np
import matplotlib.pyplot as plt
import pydicom as dcm
from rt_utils import RTStructBuilder
import os
import SimpleITK as sitk

class Plan():
    def __init__(self, parent_folder):
        """
        Class for CT or CBCT plan. This class will store the info for the different items needed to compare two dosemaps

        Args:
            parent_folder (string): Folder containing the CT slices, RTStruct, RTDose and RTPlan files
        """
        files = os.listdir(parent_folder)

        # Define the path for RTDose, RTStruct and Image slices
        for file in files:
            if "RS" in file:
                self.rtStruct_path = os.path.join(parent_folder,file)
            elif "RD" in file:
                self.rtDose_path = os.path.join(parent_folder,file)
            elif "Slices" in file:
                self.image_path = os.path.join(parent_folder,file)
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
############################################################## M A I N ###############################################

# Step 1: Create CT and CBCT objectes
ctPlan = Plan(".\\Input\\CT")
cbctPlan = Plan(".\\Input\\CBCT")

# Step 2: Register and resample CBCT Dosemap to CT Dosemap
tx_iso = iso_translation(cbctPlan.isocenter, ctPlan.isocenter)
cbct_dose_resampled = resampleDose_to_refGrid(cbctPlan, ctPlan, tx_iso)

# Save the resampled cbct dose map as an image
sitk.WriteImage(cbct_dose_resampled, "cbct_dose_resampled.nii.gz")

# Step 3: Subtract the resampled cbct dose map and ct dose map
resampled_cbct_dose_arr = sitk.GetArrayFromImage(cbct_dose_resampled) * cbctPlan.doseGridScaling
ct_dose_arr = ctPlan.rtDose_arr

dose_diff_arr = subtract_dosemaps(resampled_cbct_dose_arr,ct_dose_arr)

# Save the difference dose map as an image
dose_diff_image = sitk.GetImageFromArray(dose_diff_arr)
dose_diff_image.CopyInformation(ctPlan.rtDose_sitk)
sitk.WriteImage(dose_diff_image, "dose_difference.nii.gz")

# Step 5: Calculate the error in each structure

for structure in (s for s in ctPlan.roi_names if "Couch" not in s and "ORFIT" not in s) :
    print(structure)

pass

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

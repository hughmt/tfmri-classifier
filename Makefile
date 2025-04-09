# S3 base path and local destination
S3_BASE = s3://openneuro.org/ds002785/derivatives/fmriprep
DEST_DIR = data/ds002785/derivatives/fmriprep

# Tasks to support
TASKS = workingmemory emomatching faces gstroop restingstate anticipation

# Ensure local dir exists
PREPARE = mkdir -p $(DEST_DIR)

# Task sync template (pulls only necessary files for each task from all subjects)
define SYNC_TASK_template
$(1):
	$(PREPARE) && aws s3 sync --no-sign-request $(S3_BASE)/ $(DEST_DIR)/ \
		--exclude "*" \
		--include "sub-*/sub-*_task-$(1)_*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" \
		--include "sub-*/sub-*_task-$(1)_*_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz" \
		--include "sub-*/sub-*_task-$(1)_*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz" \
		--include "sub-*/sub-*_task-$(1)_*desc-confounds_regressors.tsv"
endef

# Generate rules for each task
$(foreach task,$(TASKS),$(eval $(call SYNC_TASK_template,$(task))))

# Download all tasks across all subjects
all:
	@$(foreach task,$(TASKS),$(MAKE) $(task);)

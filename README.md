# wildme-segmentation-pytorch

## Installation

```bash
./run_developer_setup.sh
```

Also make sure to include the plugin in `/wbia/wildbook-ia/wbia/control/IBEISControl.py` file (https://github.com/WildMeOrg/wildbook-ia), by adding the following block of code at the top of the file:

```python
if ut.get_argflag('--segmentation'):
    AUTOLOAD_PLUGIN_MODNAMES += [
        (('--no-segmentation', '--nosegmentation'), 'wbia_segmentation._plugin'),
    ]
```

The re-install wbia package:

```bash
./run_developer_setup.sh
```

## Python API

Starting from a wbia running container:

1. Make sure you define a config file, like the one defined in `wbia-plugin-segmentationwbia_segmentation/configs/01_snowleopard_segformer.yaml`. Where the relevant parameters for running inference against a species DB are:

* `mask_suffix`: Suffix for mask image file name and file extension (.png, .jpg, etc.).
* `inference_mask_dir`: Directory to save mask image files.
* `img_height`: Height of resulting mask images.
* `img_width`: Width of resulting mask images.
* `transforms_inference`: PyTorch transformation (string) or transformations (list) that should be applied to images before model prediction. (default to 'resize').
* `path_to_model`: Path of trained/finetuned model wights.
* `batch_size`: How many samples to process in parallel.

2. Start iPython with segmentation plugin loaded and ibs instance created.

```bash
$ embed --segmentation
```

3. Load aids from database and get cached result (if segmentation masks have already been computed) else new segmentation masks are generated and saved into cache table.

```python
aid_list = ibs.get_valid_aids()
ibs.depc_annot.get('SegmentationMask', aid_list, 'seg_mask', {"config_path": 'snowleopard'})
ibs.depc_annot.delete_property('SegmentationMask', gid_list, {"config_path": 'snowleopard'})
```

You can also call `register_segmentations` method to generate segmentation masks for all DB images specified without cache saving

```python
aid_list = ibs.get_valid_aids()
seg_mask_list = ibs.register_segmentations(aid_list, config_url='snowleopard')
```

The method `_render_segmentations` can be used for visualization purposes since it will save the images with the mask overlayed on top of the images.

```python
aid_list = ibs.get_valid_aids()
gpath_list, names_list, seg_mask_list = ibs._render_segmentations(aid_list, config_url='snowleopard')
```

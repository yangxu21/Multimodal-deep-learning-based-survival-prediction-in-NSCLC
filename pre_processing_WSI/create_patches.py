import os
import numpy as np
import time
import pdb
import pandas as pd

from wsi_core.batch_process_utils import initialize_df
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchPatches


def stitching(file_path, downscale = 64):
	start = time.time()
	heatmap = StitchPatches(file_path, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

def segment(WSI_object, seg_params, filter_params):
	### Start Seg Timer
	start_time = time.time()

	# Segment
	WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	### Start Patch Timer
	start_time = time.time()

	# Patch
	file_path = WSI_object.createPatches_bag_hdf5(**kwargs, save_coord=True)

	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed

def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
				  patch_size = 256, step_size = 256, custom_downsample=1, 
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8 }, 
				  vis_params = {'vis_level': -1, 'line_thickness': 250},
				  patch_params = {'white_thresh': 5, 'black_thresh': 40, 'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, auto_skip=True, process_list = None):
	
	# remove files that start with '.'
	slides = sorted(os.listdir(source))
	slides = [slide for slide in slides if not slide.startswith('.')]
	slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]


	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params, save_patches=True)
	
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params, save_patches=True)


	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)
	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in range(total):
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0
		slide_id, _ = os.path.splitext(slide)

		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

	
		# Inialize WSI
		full_path = os.path.join(source, slide)
		WSI_object = WholeSlideImage(full_path)

		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
			
		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}
			for key in vis_params.keys():
				current_vis_params.update({key: df.loc[idx, key]})

			for key in filter_params.keys():
				current_filter_params.update({key: df.loc[idx, key]})

			for key in seg_params.keys():
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})
		

		current_vis_params['vis_level'] = 0
		current_seg_params['seg_level'] = 0

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = WSI_object.wsi_resize.size 
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		if not process_list:
			df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
			df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

		seg_time_elapsed = -1
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

		if save_mask:
			mask = WSI_object.visWSI(**current_vis_params)
			mask_path = os.path.join(mask_save_dir, slide_id+'.png')
			mask.save(mask_path)

		patch_time_elapsed = -1 # Default time
		if patch:
			current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 
										 'save_path': patch_save_dir, 'custom_downsample': custom_downsample})
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object, **current_patch_params)
		
		stitch_time_elapsed = -1
		if stitch:
			try:
				file_path = os.path.join(patch_save_dir, slide_id+'.h5')
				heatmap, stitch_time_elapsed = stitching(file_path, downscale=64)
				stitch_path = os.path.join(stitch_save_dir, slide_id+'.png')
				heatmap.save(stitch_path)
			except:
				print('stitching failed')
				continue

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))
		
	return seg_times, patch_times


source = 'resized_image_file'
save_dir = 'patch_dir'
patch_save_dir = os.path.join(save_dir, 'patches')
mask_save_dir = os.path.join(save_dir, 'masks')
stitch_save_dir = os.path.join(save_dir, 'stitches')
# make directories
os.makedirs(patch_save_dir, exist_ok=True)
os.makedirs(mask_save_dir, exist_ok=True)
os.makedirs(stitch_save_dir, exist_ok=True)

seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
				patch_size = 256, step_size = 256, custom_downsample=1, 
				seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				'keep_ids': 'none', 'exclude_ids': 'none'},
				filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8 }, 
				vis_params = {'vis_level': -1, 'line_thickness': 250},
				patch_params = {'white_thresh': 5, 'black_thresh': 40, 'use_padding': True, 'contour_fn': 'four_pt'},
				patch_level = 0,
				use_default_params = False, 
				seg = True, save_mask = True, 
				stitch= False, 
				patch = True, auto_skip=True, process_list = None)


To generate necessary files, place `bbs.txt` in this directory and then run:

```
python select_short_building_blocks.py --filename bbs.txt
python subsample_building_blocks.py --random True --n 10000 --filename short_building_blocks.txt
python sanitize_building_blocks.py --building_blocks_filename short_building_blocks_subsampled_10000.txt --output_filename sanitised_bbs.txt
python remove_duplicates.py --building_blocks_filename sanitised_bbs.txt --output_filename enamine_bbs.txt
python precompute_bb_masks.py
```

Also copy `enamine_bbs_costs.sdf` into this directory and set line 19 of `tasks/config.py` in the main synflownet repo to have `"enamine_bbs_costs.sdf"`
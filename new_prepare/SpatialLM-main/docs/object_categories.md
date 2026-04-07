# SpatialLM Object Categories

SpatialLM supports detection of 59 furniture and object categories for indoor scenes. Here is the complete list:

1. sofa
2. chair
3. dining_chair
4. bar_chair
5. stool
6. bed
7. pillow
8. wardrobe
9. nightstand
10. tv_cabinet
11. wine_cabinet
12. bathroom_cabinet
13. shoe_cabinet
14. entrance_cabinet
15. decorative_cabinet
16. washing_cabinet
17. wall_cabinet
18. sideboard
19. cupboard
20. coffee_table
21. dining_table
22. side_table
23. dressing_table
24. desk
25. integrated_stove
26. gas_stove
27. range_hood
28. micro-wave_oven
29. sink
30. stove
31. refrigerator
32. hand_sink
33. shower
34. shower_room
35. toilet
36. tub
37. illumination
38. chandelier
39. floor-standing_lamp
40. wall_decoration
41. painting
42. curtain
43. carpet
44. plants
45. potted_bonsai
46. tv
47. computer
48. air_conditioner
49. washing_machine
50. clothes_rack
51. mirror
52. bookcase
53. cushion
54. bar
55. screen
56. combination_sofa
57. dining_table_combination
58. leisure_table_and_chair_combination
59. multifunctional_combination_bed

## Usage Example

To detect specific categories, use the `--category` parameter with the inference script:

```bash
python inference.py --point_cloud pcd/scene0000_00.ply --output scene0000_00.txt --model_path manycore-research/SpatialLM1.1-Qwen-0.5B --detect_type object --category bed nightstand
```
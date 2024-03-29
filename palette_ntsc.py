import numpy as np

# Indexed by (trailing 8-bit dot pattern, x % 4)
SRGB = {
  (0, 0): np.array((0, 0, 0)),
  (1, 0): np.array((0, 37, 0)),
  (2, 0): np.array((13, 15, 0)),
  (3, 0): np.array((0, 53, 0)),
  (4, 0): np.array((64, 0, 24)),
  (5, 0): np.array((0, 0, 0)),
  (6, 0): np.array((78, 0, 0)),
  (7, 0): np.array((13, 15, 0)),
  (8, 0): np.array((0, 0, 115)),
  (9, 0): np.array((0, 22, 90)),
  (10, 0): np.array((0, 0, 0)),
  (11, 0): np.array((0, 37, 0)),
  (12, 0): np.array((50, 0, 139)),
  (13, 0): np.array((0, 0, 115)),
  (14, 0): np.array((64, 0, 24)),
  (15, 0): np.array((0, 0, 0)),
  (16, 0): np.array((0, 101, 39)),
  (17, 0): np.array((0, 139, 14)),
  (18, 0): np.array((12, 116, 0)),
  (19, 0): np.array((0, 154, 0)),
  (20, 0): np.array((63, 63, 63)),
  (21, 0): np.array((0, 101, 39)),
  (22, 0): np.array((77, 79, 0)),
  (23, 0): np.array((12, 116, 0)),
  (24, 0): np.array((0, 86, 154)),
  (25, 0): np.array((0, 123, 130)),
  (26, 0): np.array((0, 101, 39)),
  (27, 0): np.array((0, 139, 14)),
  (28, 0): np.array((49, 48, 179)),
  (29, 0): np.array((0, 86, 154)),
  (30, 0): np.array((63, 63, 63)),
  (31, 0): np.array((0, 101, 39)),
  (32, 0): np.array((77, 79, 0)),
  (33, 0): np.array((12, 116, 0)),
  (34, 0): np.array((91, 94, 0)),
  (35, 0): np.array((26, 132, 0)),
  (36, 0): np.array((142, 41, 0)),
  (37, 0): np.array((77, 79, 0)),
  (38, 0): np.array((155, 56, 0)),
  (39, 0): np.array((91, 94, 0)),
  (40, 0): np.array((63, 63, 63)),
  (41, 0): np.array((0, 101, 39)),
  (42, 0): np.array((77, 79, 0)),
  (43, 0): np.array((12, 116, 0)),
  (44, 0): np.array((128, 26, 88)),
  (45, 0): np.array((63, 63, 63)),
  (46, 0): np.array((142, 41, 0)),
  (47, 0): np.array((77, 79, 0)),
  (48, 0): np.array((76, 180, 0)),
  (49, 0): np.array((11, 218, 0)),
  (50, 0): np.array((90, 196, 0)),
  (51, 0): np.array((25, 233, 0)),
  (52, 0): np.array((141, 142, 12)),
  (53, 0): np.array((76, 180, 0)),
  (54, 0): np.array((155, 158, 0)),
  (55, 0): np.array((90, 196, 0)),
  (56, 0): np.array((62, 165, 102)),
  (57, 0): np.array((0, 202, 78)),
  (58, 0): np.array((76, 180, 0)),
  (59, 0): np.array((11, 218, 0)),
  (60, 0): np.array((127, 127, 127)),
  (61, 0): np.array((62, 165, 102)),
  (62, 0): np.array((141, 142, 12)),
  (63, 0): np.array((76, 180, 0)),
  (64, 0): np.array((128, 26, 88)),
  (65, 0): np.array((63, 63, 63)),
  (66, 0): np.array((142, 41, 0)),
  (67, 0): np.array((77, 79, 0)),
  (68, 0): np.array((193, 0, 112)),
  (69, 0): np.array((128, 26, 88)),
  (70, 0): np.array((206, 3, 0)),
  (71, 0): np.array((142, 41, 0)),
  (72, 0): np.array((114, 10, 203)),
  (73, 0): np.array((49, 48, 179)),
  (74, 0): np.array((128, 26, 88)),
  (75, 0): np.array((63, 63, 63)),
  (76, 0): np.array((179, 0, 228)),
  (77, 0): np.array((114, 10, 203)),
  (78, 0): np.array((193, 0, 112)),
  (79, 0): np.array((128, 26, 88)),
  (80, 0): np.array((127, 127, 127)),
  (81, 0): np.array((62, 165, 102)),
  (82, 0): np.array((141, 142, 12)),
  (83, 0): np.array((76, 180, 0)),
  (84, 0): np.array((192, 89, 152)),
  (85, 0): np.array((127, 127, 127)),
  (86, 0): np.array((205, 105, 36)),
  (87, 0): np.array((141, 142, 12)),
  (88, 0): np.array((113, 112, 242)),
  (89, 0): np.array((49, 149, 218)),
  (90, 0): np.array((127, 127, 127)),
  (91, 0): np.array((62, 165, 102)),
  (92, 0): np.array((178, 74, 255)),
  (93, 0): np.array((113, 112, 242)),
  (94, 0): np.array((192, 89, 152)),
  (95, 0): np.array((127, 127, 127)),
  (96, 0): np.array((205, 105, 36)),
  (97, 0): np.array((141, 142, 12)),
  (98, 0): np.array((219, 120, 0)),
  (99, 0): np.array((155, 158, 0)),
  (100, 0): np.array((255, 67, 61)),
  (101, 0): np.array((205, 105, 36)),
  (102, 0): np.array((255, 82, 0)),
  (103, 0): np.array((219, 120, 0)),
  (104, 0): np.array((192, 89, 152)),
  (105, 0): np.array((127, 127, 127)),
  (106, 0): np.array((205, 105, 36)),
  (107, 0): np.array((141, 142, 12)),
  (108, 0): np.array((255, 52, 176)),
  (109, 0): np.array((192, 89, 152)),
  (110, 0): np.array((255, 67, 61)),
  (111, 0): np.array((205, 105, 36)),
  (112, 0): np.array((205, 206, 75)),
  (113, 0): np.array((140, 244, 51)),
  (114, 0): np.array((218, 222, 0)),
  (115, 0): np.array((154, 255, 0)),
  (116, 0): np.array((255, 168, 100)),
  (117, 0): np.array((205, 206, 75)),
  (118, 0): np.array((255, 184, 0)),
  (119, 0): np.array((218, 222, 0)),
  (120, 0): np.array((191, 191, 191)),
  (121, 0): np.array((126, 228, 166)),
  (122, 0): np.array((205, 206, 75)),
  (123, 0): np.array((140, 244, 51)),
  (124, 0): np.array((255, 153, 215)),
  (125, 0): np.array((191, 191, 191)),
  (126, 0): np.array((255, 168, 100)),
  (127, 0): np.array((205, 206, 75)),
  (128, 0): np.array((49, 48, 179)),
  (129, 0): np.array((0, 86, 154)),
  (130, 0): np.array((63, 63, 63)),
  (131, 0): np.array((0, 101, 39)),
  (132, 0): np.array((114, 10, 203)),
  (133, 0): np.array((49, 48, 179)),
  (134, 0): np.array((128, 26, 88)),
  (135, 0): np.array((63, 63, 63)),
  (136, 0): np.array((36, 32, 255)),
  (137, 0): np.array((0, 70, 255)),
  (138, 0): np.array((49, 48, 179)),
  (139, 0): np.array((0, 86, 154)),
  (140, 0): np.array((100, 0, 255)),
  (141, 0): np.array((36, 32, 255)),
  (142, 0): np.array((114, 10, 203)),
  (143, 0): np.array((49, 48, 179)),
  (144, 0): np.array((49, 149, 218)),
  (145, 0): np.array((0, 187, 193)),
  (146, 0): np.array((62, 165, 102)),
  (147, 0): np.array((0, 202, 78)),
  (148, 0): np.array((113, 112, 242)),
  (149, 0): np.array((49, 149, 218)),
  (150, 0): np.array((127, 127, 127)),
  (151, 0): np.array((62, 165, 102)),
  (152, 0): np.array((35, 134, 255)),
  (153, 0): np.array((0, 172, 255)),
  (154, 0): np.array((49, 149, 218)),
  (155, 0): np.array((0, 187, 193)),
  (156, 0): np.array((99, 96, 255)),
  (157, 0): np.array((35, 134, 255)),
  (158, 0): np.array((113, 112, 242)),
  (159, 0): np.array((49, 149, 218)),
  (160, 0): np.array((127, 127, 127)),
  (161, 0): np.array((62, 165, 102)),
  (162, 0): np.array((141, 142, 12)),
  (163, 0): np.array((76, 180, 0)),
  (164, 0): np.array((192, 89, 152)),
  (165, 0): np.array((127, 127, 127)),
  (166, 0): np.array((205, 105, 36)),
  (167, 0): np.array((141, 142, 12)),
  (168, 0): np.array((113, 112, 242)),
  (169, 0): np.array((49, 149, 218)),
  (170, 0): np.array((127, 127, 127)),
  (171, 0): np.array((62, 165, 102)),
  (172, 0): np.array((178, 74, 255)),
  (173, 0): np.array((113, 112, 242)),
  (174, 0): np.array((192, 89, 152)),
  (175, 0): np.array((127, 127, 127)),
  (176, 0): np.array((126, 228, 166)),
  (177, 0): np.array((61, 255, 142)),
  (178, 0): np.array((140, 244, 51)),
  (179, 0): np.array((75, 255, 26)),
  (180, 0): np.array((191, 191, 191)),
  (181, 0): np.array((126, 228, 166)),
  (182, 0): np.array((205, 206, 75)),
  (183, 0): np.array((140, 244, 51)),
  (184, 0): np.array((112, 213, 255)),
  (185, 0): np.array((48, 251, 255)),
  (186, 0): np.array((126, 228, 166)),
  (187, 0): np.array((61, 255, 142)),
  (188, 0): np.array((177, 175, 255)),
  (189, 0): np.array((112, 213, 255)),
  (190, 0): np.array((191, 191, 191)),
  (191, 0): np.array((126, 228, 166)),
  (192, 0): np.array((178, 74, 255)),
  (193, 0): np.array((113, 112, 242)),
  (194, 0): np.array((192, 89, 152)),
  (195, 0): np.array((127, 127, 127)),
  (196, 0): np.array((243, 36, 255)),
  (197, 0): np.array((178, 74, 255)),
  (198, 0): np.array((255, 52, 176)),
  (199, 0): np.array((192, 89, 152)),
  (200, 0): np.array((164, 58, 255)),
  (201, 0): np.array((99, 96, 255)),
  (202, 0): np.array((178, 74, 255)),
  (203, 0): np.array((113, 112, 242)),
  (204, 0): np.array((229, 21, 255)),
  (205, 0): np.array((164, 58, 255)),
  (206, 0): np.array((243, 36, 255)),
  (207, 0): np.array((178, 74, 255)),
  (208, 0): np.array((177, 175, 255)),
  (209, 0): np.array((112, 213, 255)),
  (210, 0): np.array((191, 191, 191)),
  (211, 0): np.array((126, 228, 166)),
  (212, 0): np.array((242, 138, 255)),
  (213, 0): np.array((177, 175, 255)),
  (214, 0): np.array((255, 153, 215)),
  (215, 0): np.array((191, 191, 191)),
  (216, 0): np.array((163, 160, 255)),
  (217, 0): np.array((99, 198, 255)),
  (218, 0): np.array((177, 175, 255)),
  (219, 0): np.array((112, 213, 255)),
  (220, 0): np.array((228, 122, 255)),
  (221, 0): np.array((163, 160, 255)),
  (222, 0): np.array((242, 138, 255)),
  (223, 0): np.array((177, 175, 255)),
  (224, 0): np.array((255, 153, 215)),
  (225, 0): np.array((191, 191, 191)),
  (226, 0): np.array((255, 168, 100)),
  (227, 0): np.array((205, 206, 75)),
  (228, 0): np.array((255, 115, 240)),
  (229, 0): np.array((255, 153, 215)),
  (230, 0): np.array((255, 131, 124)),
  (231, 0): np.array((255, 168, 100)),
  (232, 0): np.array((242, 138, 255)),
  (233, 0): np.array((177, 175, 255)),
  (234, 0): np.array((255, 153, 215)),
  (235, 0): np.array((191, 191, 191)),
  (236, 0): np.array((255, 100, 255)),
  (237, 0): np.array((242, 138, 255)),
  (238, 0): np.array((255, 115, 240)),
  (239, 0): np.array((255, 153, 215)),
  (240, 0): np.array((255, 255, 254)),
  (241, 0): np.array((190, 255, 230)),
  (242, 0): np.array((255, 255, 139)),
  (243, 0): np.array((204, 255, 115)),
  (244, 0): np.array((255, 217, 255)),
  (245, 0): np.array((255, 255, 254)),
  (246, 0): np.array((255, 232, 164)),
  (247, 0): np.array((255, 255, 139)),
  (248, 0): np.array((241, 239, 255)),
  (249, 0): np.array((176, 255, 255)),
  (250, 0): np.array((254, 255, 254)),
  (251, 0): np.array((190, 255, 230)),
  (252, 0): np.array((255, 201, 255)),
  (253, 0): np.array((241, 239, 255)),
  (254, 0): np.array((255, 217, 255)),
  (255, 0): np.array((255, 255, 254)),
  (0, 1): np.array((0, 0, 0)),
  (1, 1): np.array((13, 15, 0)),
  (2, 1): np.array((64, 0, 24)),
  (3, 1): np.array((78, 0, 0)),
  (4, 1): np.array((0, 0, 115)),
  (5, 1): np.array((0, 0, 0)),
  (6, 1): np.array((50, 0, 139)),
  (7, 1): np.array((64, 0, 24)),
  (8, 1): np.array((0, 37, 0)),
  (9, 1): np.array((0, 53, 0)),
  (10, 1): np.array((0, 0, 0)),
  (11, 1): np.array((13, 15, 0)),
  (12, 1): np.array((0, 22, 90)),
  (13, 1): np.array((0, 37, 0)),
  (14, 1): np.array((0, 0, 115)),
  (15, 1): np.array((0, 0, 0)),
  (16, 1): np.array((77, 79, 0)),
  (17, 1): np.array((91, 94, 0)),
  (18, 1): np.array((142, 41, 0)),
  (19, 1): np.array((155, 56, 0)),
  (20, 1): np.array((63, 63, 63)),
  (21, 1): np.array((77, 79, 0)),
  (22, 1): np.array((128, 26, 88)),
  (23, 1): np.array((142, 41, 0)),
  (24, 1): np.array((12, 116, 0)),
  (25, 1): np.array((26, 132, 0)),
  (26, 1): np.array((77, 79, 0)),
  (27, 1): np.array((91, 94, 0)),
  (28, 1): np.array((0, 101, 39)),
  (29, 1): np.array((12, 116, 0)),
  (30, 1): np.array((63, 63, 63)),
  (31, 1): np.array((77, 79, 0)),
  (32, 1): np.array((128, 26, 88)),
  (33, 1): np.array((142, 41, 0)),
  (34, 1): np.array((193, 0, 112)),
  (35, 1): np.array((206, 3, 0)),
  (36, 1): np.array((114, 10, 203)),
  (37, 1): np.array((128, 26, 88)),
  (38, 1): np.array((179, 0, 228)),
  (39, 1): np.array((193, 0, 112)),
  (40, 1): np.array((63, 63, 63)),
  (41, 1): np.array((77, 79, 0)),
  (42, 1): np.array((128, 26, 88)),
  (43, 1): np.array((142, 41, 0)),
  (44, 1): np.array((49, 48, 179)),
  (45, 1): np.array((63, 63, 63)),
  (46, 1): np.array((114, 10, 203)),
  (47, 1): np.array((128, 26, 88)),
  (48, 1): np.array((205, 105, 36)),
  (49, 1): np.array((219, 120, 0)),
  (50, 1): np.array((255, 67, 61)),
  (51, 1): np.array((255, 82, 0)),
  (52, 1): np.array((192, 89, 152)),
  (53, 1): np.array((205, 105, 36)),
  (54, 1): np.array((255, 52, 176)),
  (55, 1): np.array((255, 67, 61)),
  (56, 1): np.array((141, 142, 12)),
  (57, 1): np.array((155, 158, 0)),
  (58, 1): np.array((205, 105, 36)),
  (59, 1): np.array((219, 120, 0)),
  (60, 1): np.array((127, 127, 127)),
  (61, 1): np.array((141, 142, 12)),
  (62, 1): np.array((192, 89, 152)),
  (63, 1): np.array((205, 105, 36)),
  (64, 1): np.array((49, 48, 179)),
  (65, 1): np.array((63, 63, 63)),
  (66, 1): np.array((114, 10, 203)),
  (67, 1): np.array((128, 26, 88)),
  (68, 1): np.array((36, 32, 255)),
  (69, 1): np.array((49, 48, 179)),
  (70, 1): np.array((100, 0, 255)),
  (71, 1): np.array((114, 10, 203)),
  (72, 1): np.array((0, 86, 154)),
  (73, 1): np.array((0, 101, 39)),
  (74, 1): np.array((49, 48, 179)),
  (75, 1): np.array((63, 63, 63)),
  (76, 1): np.array((0, 70, 255)),
  (77, 1): np.array((0, 86, 154)),
  (78, 1): np.array((36, 32, 255)),
  (79, 1): np.array((49, 48, 179)),
  (80, 1): np.array((127, 127, 127)),
  (81, 1): np.array((141, 142, 12)),
  (82, 1): np.array((192, 89, 152)),
  (83, 1): np.array((205, 105, 36)),
  (84, 1): np.array((113, 112, 242)),
  (85, 1): np.array((127, 127, 127)),
  (86, 1): np.array((178, 74, 255)),
  (87, 1): np.array((192, 89, 152)),
  (88, 1): np.array((62, 165, 102)),
  (89, 1): np.array((76, 180, 0)),
  (90, 1): np.array((127, 127, 127)),
  (91, 1): np.array((141, 142, 12)),
  (92, 1): np.array((49, 149, 218)),
  (93, 1): np.array((62, 165, 102)),
  (94, 1): np.array((113, 112, 242)),
  (95, 1): np.array((127, 127, 127)),
  (96, 1): np.array((178, 74, 255)),
  (97, 1): np.array((192, 89, 152)),
  (98, 1): np.array((243, 36, 255)),
  (99, 1): np.array((255, 52, 176)),
  (100, 1): np.array((164, 58, 255)),
  (101, 1): np.array((178, 74, 255)),
  (102, 1): np.array((229, 21, 255)),
  (103, 1): np.array((243, 36, 255)),
  (104, 1): np.array((113, 112, 242)),
  (105, 1): np.array((127, 127, 127)),
  (106, 1): np.array((178, 74, 255)),
  (107, 1): np.array((192, 89, 152)),
  (108, 1): np.array((99, 96, 255)),
  (109, 1): np.array((113, 112, 242)),
  (110, 1): np.array((164, 58, 255)),
  (111, 1): np.array((178, 74, 255)),
  (112, 1): np.array((255, 153, 215)),
  (113, 1): np.array((255, 168, 100)),
  (114, 1): np.array((255, 115, 240)),
  (115, 1): np.array((255, 131, 124)),
  (116, 1): np.array((242, 138, 255)),
  (117, 1): np.array((255, 153, 215)),
  (118, 1): np.array((255, 100, 255)),
  (119, 1): np.array((255, 115, 240)),
  (120, 1): np.array((191, 191, 191)),
  (121, 1): np.array((205, 206, 75)),
  (122, 1): np.array((255, 153, 215)),
  (123, 1): np.array((255, 168, 100)),
  (124, 1): np.array((177, 175, 255)),
  (125, 1): np.array((191, 191, 191)),
  (126, 1): np.array((242, 138, 255)),
  (127, 1): np.array((255, 153, 215)),
  (128, 1): np.array((0, 101, 39)),
  (129, 1): np.array((12, 116, 0)),
  (130, 1): np.array((63, 63, 63)),
  (131, 1): np.array((77, 79, 0)),
  (132, 1): np.array((0, 86, 154)),
  (133, 1): np.array((0, 101, 39)),
  (134, 1): np.array((49, 48, 179)),
  (135, 1): np.array((63, 63, 63)),
  (136, 1): np.array((0, 139, 14)),
  (137, 1): np.array((0, 154, 0)),
  (138, 1): np.array((0, 101, 39)),
  (139, 1): np.array((12, 116, 0)),
  (140, 1): np.array((0, 123, 130)),
  (141, 1): np.array((0, 139, 14)),
  (142, 1): np.array((0, 86, 154)),
  (143, 1): np.array((0, 101, 39)),
  (144, 1): np.array((76, 180, 0)),
  (145, 1): np.array((90, 196, 0)),
  (146, 1): np.array((141, 142, 12)),
  (147, 1): np.array((155, 158, 0)),
  (148, 1): np.array((62, 165, 102)),
  (149, 1): np.array((76, 180, 0)),
  (150, 1): np.array((127, 127, 127)),
  (151, 1): np.array((141, 142, 12)),
  (152, 1): np.array((11, 218, 0)),
  (153, 1): np.array((25, 233, 0)),
  (154, 1): np.array((76, 180, 0)),
  (155, 1): np.array((90, 196, 0)),
  (156, 1): np.array((0, 202, 78)),
  (157, 1): np.array((11, 218, 0)),
  (158, 1): np.array((62, 165, 102)),
  (159, 1): np.array((76, 180, 0)),
  (160, 1): np.array((127, 127, 127)),
  (161, 1): np.array((141, 142, 12)),
  (162, 1): np.array((192, 89, 152)),
  (163, 1): np.array((205, 105, 36)),
  (164, 1): np.array((113, 112, 242)),
  (165, 1): np.array((127, 127, 127)),
  (166, 1): np.array((178, 74, 255)),
  (167, 1): np.array((192, 89, 152)),
  (168, 1): np.array((62, 165, 102)),
  (169, 1): np.array((76, 180, 0)),
  (170, 1): np.array((127, 127, 127)),
  (171, 1): np.array((141, 142, 12)),
  (172, 1): np.array((49, 149, 218)),
  (173, 1): np.array((62, 165, 102)),
  (174, 1): np.array((113, 112, 242)),
  (175, 1): np.array((127, 127, 127)),
  (176, 1): np.array((205, 206, 75)),
  (177, 1): np.array((218, 222, 0)),
  (178, 1): np.array((255, 168, 100)),
  (179, 1): np.array((255, 184, 0)),
  (180, 1): np.array((191, 191, 191)),
  (181, 1): np.array((205, 206, 75)),
  (182, 1): np.array((255, 153, 215)),
  (183, 1): np.array((255, 168, 100)),
  (184, 1): np.array((140, 244, 51)),
  (185, 1): np.array((154, 255, 0)),
  (186, 1): np.array((205, 206, 75)),
  (187, 1): np.array((218, 222, 0)),
  (188, 1): np.array((126, 228, 166)),
  (189, 1): np.array((140, 244, 51)),
  (190, 1): np.array((191, 191, 191)),
  (191, 1): np.array((205, 206, 75)),
  (192, 1): np.array((49, 149, 218)),
  (193, 1): np.array((62, 165, 102)),
  (194, 1): np.array((113, 112, 242)),
  (195, 1): np.array((127, 127, 127)),
  (196, 1): np.array((35, 134, 255)),
  (197, 1): np.array((49, 149, 218)),
  (198, 1): np.array((99, 96, 255)),
  (199, 1): np.array((113, 112, 242)),
  (200, 1): np.array((0, 187, 193)),
  (201, 1): np.array((0, 202, 78)),
  (202, 1): np.array((49, 149, 218)),
  (203, 1): np.array((62, 165, 102)),
  (204, 1): np.array((0, 172, 255)),
  (205, 1): np.array((0, 187, 193)),
  (206, 1): np.array((35, 134, 255)),
  (207, 1): np.array((49, 149, 218)),
  (208, 1): np.array((126, 228, 166)),
  (209, 1): np.array((140, 244, 51)),
  (210, 1): np.array((191, 191, 191)),
  (211, 1): np.array((205, 206, 75)),
  (212, 1): np.array((112, 213, 255)),
  (213, 1): np.array((126, 228, 166)),
  (214, 1): np.array((177, 175, 255)),
  (215, 1): np.array((191, 191, 191)),
  (216, 1): np.array((61, 255, 142)),
  (217, 1): np.array((75, 255, 26)),
  (218, 1): np.array((126, 228, 166)),
  (219, 1): np.array((140, 244, 51)),
  (220, 1): np.array((48, 251, 255)),
  (221, 1): np.array((61, 255, 142)),
  (222, 1): np.array((112, 213, 255)),
  (223, 1): np.array((126, 228, 166)),
  (224, 1): np.array((177, 175, 255)),
  (225, 1): np.array((191, 191, 191)),
  (226, 1): np.array((242, 138, 255)),
  (227, 1): np.array((255, 153, 215)),
  (228, 1): np.array((163, 160, 255)),
  (229, 1): np.array((177, 175, 255)),
  (230, 1): np.array((228, 122, 255)),
  (231, 1): np.array((242, 138, 255)),
  (232, 1): np.array((112, 213, 255)),
  (233, 1): np.array((126, 228, 166)),
  (234, 1): np.array((177, 175, 255)),
  (235, 1): np.array((191, 191, 191)),
  (236, 1): np.array((99, 198, 255)),
  (237, 1): np.array((112, 213, 255)),
  (238, 1): np.array((163, 160, 255)),
  (239, 1): np.array((177, 175, 255)),
  (240, 1): np.array((255, 255, 254)),
  (241, 1): np.array((255, 255, 139)),
  (242, 1): np.array((255, 217, 255)),
  (243, 1): np.array((255, 232, 164)),
  (244, 1): np.array((241, 239, 255)),
  (245, 1): np.array((254, 255, 254)),
  (246, 1): np.array((255, 201, 255)),
  (247, 1): np.array((255, 217, 255)),
  (248, 1): np.array((190, 255, 230)),
  (249, 1): np.array((204, 255, 115)),
  (250, 1): np.array((255, 255, 254)),
  (251, 1): np.array((255, 255, 139)),
  (252, 1): np.array((176, 255, 255)),
  (253, 1): np.array((190, 255, 230)),
  (254, 1): np.array((241, 239, 255)),
  (255, 1): np.array((254, 255, 254)),
  (0, 2): np.array((0, 0, 0)),
  (1, 2): np.array((64, 0, 24)),
  (2, 2): np.array((0, 0, 115)),
  (3, 2): np.array((50, 0, 139)),
  (4, 2): np.array((0, 37, 0)),
  (5, 2): np.array((0, 0, 0)),
  (6, 2): np.array((0, 22, 90)),
  (7, 2): np.array((0, 0, 115)),
  (8, 2): np.array((13, 15, 0)),
  (9, 2): np.array((78, 0, 0)),
  (10, 2): np.array((0, 0, 0)),
  (11, 2): np.array((64, 0, 24)),
  (12, 2): np.array((0, 53, 0)),
  (13, 2): np.array((13, 15, 0)),
  (14, 2): np.array((0, 37, 0)),
  (15, 2): np.array((0, 0, 0)),
  (16, 2): np.array((128, 26, 88)),
  (17, 2): np.array((193, 0, 112)),
  (18, 2): np.array((114, 10, 203)),
  (19, 2): np.array((179, 0, 228)),
  (20, 2): np.array((63, 63, 63)),
  (21, 2): np.array((128, 26, 88)),
  (22, 2): np.array((49, 48, 179)),
  (23, 2): np.array((114, 10, 203)),
  (24, 2): np.array((142, 41, 0)),
  (25, 2): np.array((206, 3, 0)),
  (26, 2): np.array((128, 26, 88)),
  (27, 2): np.array((193, 0, 112)),
  (28, 2): np.array((77, 79, 0)),
  (29, 2): np.array((142, 41, 0)),
  (30, 2): np.array((63, 63, 63)),
  (31, 2): np.array((128, 26, 88)),
  (32, 2): np.array((49, 48, 179)),
  (33, 2): np.array((114, 10, 203)),
  (34, 2): np.array((36, 32, 255)),
  (35, 2): np.array((100, 0, 255)),
  (36, 2): np.array((0, 86, 154)),
  (37, 2): np.array((49, 48, 179)),
  (38, 2): np.array((0, 70, 255)),
  (39, 2): np.array((36, 32, 255)),
  (40, 2): np.array((63, 63, 63)),
  (41, 2): np.array((128, 26, 88)),
  (42, 2): np.array((49, 48, 179)),
  (43, 2): np.array((114, 10, 203)),
  (44, 2): np.array((0, 101, 39)),
  (45, 2): np.array((63, 63, 63)),
  (46, 2): np.array((0, 86, 154)),
  (47, 2): np.array((49, 48, 179)),
  (48, 2): np.array((178, 74, 255)),
  (49, 2): np.array((243, 36, 255)),
  (50, 2): np.array((164, 58, 255)),
  (51, 2): np.array((229, 21, 255)),
  (52, 2): np.array((113, 112, 242)),
  (53, 2): np.array((178, 74, 255)),
  (54, 2): np.array((99, 96, 255)),
  (55, 2): np.array((164, 58, 255)),
  (56, 2): np.array((192, 89, 152)),
  (57, 2): np.array((255, 52, 176)),
  (58, 2): np.array((178, 74, 255)),
  (59, 2): np.array((243, 36, 255)),
  (60, 2): np.array((127, 127, 127)),
  (61, 2): np.array((192, 89, 152)),
  (62, 2): np.array((113, 112, 242)),
  (63, 2): np.array((178, 74, 255)),
  (64, 2): np.array((0, 101, 39)),
  (65, 2): np.array((63, 63, 63)),
  (66, 2): np.array((0, 86, 154)),
  (67, 2): np.array((49, 48, 179)),
  (68, 2): np.array((0, 139, 14)),
  (69, 2): np.array((0, 101, 39)),
  (70, 2): np.array((0, 123, 130)),
  (71, 2): np.array((0, 86, 154)),
  (72, 2): np.array((12, 116, 0)),
  (73, 2): np.array((77, 79, 0)),
  (74, 2): np.array((0, 101, 39)),
  (75, 2): np.array((63, 63, 63)),
  (76, 2): np.array((0, 154, 0)),
  (77, 2): np.array((12, 116, 0)),
  (78, 2): np.array((0, 139, 14)),
  (79, 2): np.array((0, 101, 39)),
  (80, 2): np.array((127, 127, 127)),
  (81, 2): np.array((192, 89, 152)),
  (82, 2): np.array((113, 112, 242)),
  (83, 2): np.array((178, 74, 255)),
  (84, 2): np.array((62, 165, 102)),
  (85, 2): np.array((127, 127, 127)),
  (86, 2): np.array((49, 149, 218)),
  (87, 2): np.array((113, 112, 242)),
  (88, 2): np.array((141, 142, 12)),
  (89, 2): np.array((205, 105, 36)),
  (90, 2): np.array((127, 127, 127)),
  (91, 2): np.array((192, 89, 152)),
  (92, 2): np.array((76, 180, 0)),
  (93, 2): np.array((141, 142, 12)),
  (94, 2): np.array((62, 165, 102)),
  (95, 2): np.array((127, 127, 127)),
  (96, 2): np.array((49, 149, 218)),
  (97, 2): np.array((113, 112, 242)),
  (98, 2): np.array((35, 134, 255)),
  (99, 2): np.array((99, 96, 255)),
  (100, 2): np.array((0, 187, 193)),
  (101, 2): np.array((49, 149, 218)),
  (102, 2): np.array((0, 172, 255)),
  (103, 2): np.array((35, 134, 255)),
  (104, 2): np.array((62, 165, 102)),
  (105, 2): np.array((127, 127, 127)),
  (106, 2): np.array((49, 149, 218)),
  (107, 2): np.array((113, 112, 242)),
  (108, 2): np.array((0, 202, 78)),
  (109, 2): np.array((62, 165, 102)),
  (110, 2): np.array((0, 187, 193)),
  (111, 2): np.array((49, 149, 218)),
  (112, 2): np.array((177, 175, 255)),
  (113, 2): np.array((242, 138, 255)),
  (114, 2): np.array((163, 160, 255)),
  (115, 2): np.array((228, 122, 255)),
  (116, 2): np.array((112, 213, 255)),
  (117, 2): np.array((177, 175, 255)),
  (118, 2): np.array((99, 198, 255)),
  (119, 2): np.array((163, 160, 255)),
  (120, 2): np.array((191, 191, 191)),
  (121, 2): np.array((255, 153, 215)),
  (122, 2): np.array((177, 175, 255)),
  (123, 2): np.array((242, 138, 255)),
  (124, 2): np.array((126, 228, 166)),
  (125, 2): np.array((191, 191, 191)),
  (126, 2): np.array((112, 213, 255)),
  (127, 2): np.array((177, 175, 255)),
  (128, 2): np.array((77, 79, 0)),
  (129, 2): np.array((142, 41, 0)),
  (130, 2): np.array((63, 63, 63)),
  (131, 2): np.array((128, 26, 88)),
  (132, 2): np.array((12, 116, 0)),
  (133, 2): np.array((77, 79, 0)),
  (134, 2): np.array((0, 101, 39)),
  (135, 2): np.array((63, 63, 63)),
  (136, 2): np.array((91, 94, 0)),
  (137, 2): np.array((155, 56, 0)),
  (138, 2): np.array((77, 79, 0)),
  (139, 2): np.array((142, 41, 0)),
  (140, 2): np.array((26, 132, 0)),
  (141, 2): np.array((91, 94, 0)),
  (142, 2): np.array((12, 116, 0)),
  (143, 2): np.array((77, 79, 0)),
  (144, 2): np.array((205, 105, 36)),
  (145, 2): np.array((255, 67, 61)),
  (146, 2): np.array((192, 89, 152)),
  (147, 2): np.array((255, 52, 176)),
  (148, 2): np.array((141, 142, 12)),
  (149, 2): np.array((205, 105, 36)),
  (150, 2): np.array((127, 127, 127)),
  (151, 2): np.array((192, 89, 152)),
  (152, 2): np.array((219, 120, 0)),
  (153, 2): np.array((255, 82, 0)),
  (154, 2): np.array((205, 105, 36)),
  (155, 2): np.array((255, 67, 61)),
  (156, 2): np.array((155, 158, 0)),
  (157, 2): np.array((219, 120, 0)),
  (158, 2): np.array((141, 142, 12)),
  (159, 2): np.array((205, 105, 36)),
  (160, 2): np.array((127, 127, 127)),
  (161, 2): np.array((192, 89, 152)),
  (162, 2): np.array((113, 112, 242)),
  (163, 2): np.array((178, 74, 255)),
  (164, 2): np.array((62, 165, 102)),
  (165, 2): np.array((127, 127, 127)),
  (166, 2): np.array((49, 149, 218)),
  (167, 2): np.array((113, 112, 242)),
  (168, 2): np.array((141, 142, 12)),
  (169, 2): np.array((205, 105, 36)),
  (170, 2): np.array((127, 127, 127)),
  (171, 2): np.array((192, 89, 152)),
  (172, 2): np.array((76, 180, 0)),
  (173, 2): np.array((141, 142, 12)),
  (174, 2): np.array((62, 165, 102)),
  (175, 2): np.array((127, 127, 127)),
  (176, 2): np.array((255, 153, 215)),
  (177, 2): np.array((255, 115, 240)),
  (178, 2): np.array((242, 138, 255)),
  (179, 2): np.array((255, 100, 255)),
  (180, 2): np.array((191, 191, 191)),
  (181, 2): np.array((255, 153, 215)),
  (182, 2): np.array((177, 175, 255)),
  (183, 2): np.array((242, 138, 255)),
  (184, 2): np.array((255, 168, 100)),
  (185, 2): np.array((255, 131, 124)),
  (186, 2): np.array((255, 153, 215)),
  (187, 2): np.array((255, 115, 240)),
  (188, 2): np.array((205, 206, 75)),
  (189, 2): np.array((255, 168, 100)),
  (190, 2): np.array((191, 191, 191)),
  (191, 2): np.array((255, 153, 215)),
  (192, 2): np.array((76, 180, 0)),
  (193, 2): np.array((141, 142, 12)),
  (194, 2): np.array((62, 165, 102)),
  (195, 2): np.array((127, 127, 127)),
  (196, 2): np.array((11, 218, 0)),
  (197, 2): np.array((76, 180, 0)),
  (198, 2): np.array((0, 202, 78)),
  (199, 2): np.array((62, 165, 102)),
  (200, 2): np.array((90, 196, 0)),
  (201, 2): np.array((155, 158, 0)),
  (202, 2): np.array((76, 180, 0)),
  (203, 2): np.array((141, 142, 12)),
  (204, 2): np.array((25, 233, 0)),
  (205, 2): np.array((90, 196, 0)),
  (206, 2): np.array((11, 218, 0)),
  (207, 2): np.array((76, 180, 0)),
  (208, 2): np.array((205, 206, 75)),
  (209, 2): np.array((255, 168, 100)),
  (210, 2): np.array((191, 191, 191)),
  (211, 2): np.array((255, 153, 215)),
  (212, 2): np.array((140, 244, 51)),
  (213, 2): np.array((205, 206, 75)),
  (214, 2): np.array((126, 228, 166)),
  (215, 2): np.array((191, 191, 191)),
  (216, 2): np.array((218, 222, 0)),
  (217, 2): np.array((255, 184, 0)),
  (218, 2): np.array((205, 206, 75)),
  (219, 2): np.array((255, 168, 100)),
  (220, 2): np.array((154, 255, 0)),
  (221, 2): np.array((218, 222, 0)),
  (222, 2): np.array((140, 244, 51)),
  (223, 2): np.array((205, 206, 75)),
  (224, 2): np.array((126, 228, 166)),
  (225, 2): np.array((191, 191, 191)),
  (226, 2): np.array((112, 213, 255)),
  (227, 2): np.array((177, 175, 255)),
  (228, 2): np.array((61, 255, 142)),
  (229, 2): np.array((126, 228, 166)),
  (230, 2): np.array((48, 251, 255)),
  (231, 2): np.array((112, 213, 255)),
  (232, 2): np.array((140, 244, 51)),
  (233, 2): np.array((205, 206, 75)),
  (234, 2): np.array((126, 228, 166)),
  (235, 2): np.array((191, 191, 191)),
  (236, 2): np.array((75, 255, 26)),
  (237, 2): np.array((140, 244, 51)),
  (238, 2): np.array((61, 255, 142)),
  (239, 2): np.array((126, 228, 166)),
  (240, 2): np.array((255, 255, 254)),
  (241, 2): np.array((255, 217, 255)),
  (242, 2): np.array((241, 239, 255)),
  (243, 2): np.array((255, 201, 255)),
  (244, 2): np.array((190, 255, 230)),
  (245, 2): np.array((255, 255, 254)),
  (246, 2): np.array((176, 255, 255)),
  (247, 2): np.array((241, 239, 255)),
  (248, 2): np.array((255, 255, 139)),
  (249, 2): np.array((255, 232, 164)),
  (250, 2): np.array((254, 255, 254)),
  (251, 2): np.array((255, 217, 255)),
  (252, 2): np.array((204, 255, 115)),
  (253, 2): np.array((255, 255, 139)),
  (254, 2): np.array((190, 255, 230)),
  (255, 2): np.array((254, 255, 254)),
  (0, 3): np.array((0, 0, 0)),
  (1, 3): np.array((0, 0, 115)),
  (2, 3): np.array((0, 37, 0)),
  (3, 3): np.array((0, 22, 90)),
  (4, 3): np.array((13, 15, 0)),
  (5, 3): np.array((0, 0, 0)),
  (6, 3): np.array((0, 53, 0)),
  (7, 3): np.array((0, 37, 0)),
  (8, 3): np.array((64, 0, 24)),
  (9, 3): np.array((50, 0, 139)),
  (10, 3): np.array((0, 0, 0)),
  (11, 3): np.array((0, 0, 115)),
  (12, 3): np.array((78, 0, 0)),
  (13, 3): np.array((64, 0, 24)),
  (14, 3): np.array((13, 15, 0)),
  (15, 3): np.array((0, 0, 0)),
  (16, 3): np.array((49, 48, 179)),
  (17, 3): np.array((36, 32, 255)),
  (18, 3): np.array((0, 86, 154)),
  (19, 3): np.array((0, 70, 255)),
  (20, 3): np.array((63, 63, 63)),
  (21, 3): np.array((49, 48, 179)),
  (22, 3): np.array((0, 101, 39)),
  (23, 3): np.array((0, 86, 154)),
  (24, 3): np.array((114, 10, 203)),
  (25, 3): np.array((100, 0, 255)),
  (26, 3): np.array((49, 48, 179)),
  (27, 3): np.array((36, 32, 255)),
  (28, 3): np.array((128, 26, 88)),
  (29, 3): np.array((114, 10, 203)),
  (30, 3): np.array((63, 63, 63)),
  (31, 3): np.array((49, 48, 179)),
  (32, 3): np.array((0, 101, 39)),
  (33, 3): np.array((0, 86, 154)),
  (34, 3): np.array((0, 139, 14)),
  (35, 3): np.array((0, 123, 130)),
  (36, 3): np.array((12, 116, 0)),
  (37, 3): np.array((0, 101, 39)),
  (38, 3): np.array((0, 154, 0)),
  (39, 3): np.array((0, 139, 14)),
  (40, 3): np.array((63, 63, 63)),
  (41, 3): np.array((49, 48, 179)),
  (42, 3): np.array((0, 101, 39)),
  (43, 3): np.array((0, 86, 154)),
  (44, 3): np.array((77, 79, 0)),
  (45, 3): np.array((63, 63, 63)),
  (46, 3): np.array((12, 116, 0)),
  (47, 3): np.array((0, 101, 39)),
  (48, 3): np.array((49, 149, 218)),
  (49, 3): np.array((35, 134, 255)),
  (50, 3): np.array((0, 187, 193)),
  (51, 3): np.array((0, 172, 255)),
  (52, 3): np.array((62, 165, 102)),
  (53, 3): np.array((49, 149, 218)),
  (54, 3): np.array((0, 202, 78)),
  (55, 3): np.array((0, 187, 193)),
  (56, 3): np.array((113, 112, 242)),
  (57, 3): np.array((99, 96, 255)),
  (58, 3): np.array((49, 149, 218)),
  (59, 3): np.array((35, 134, 255)),
  (60, 3): np.array((127, 127, 127)),
  (61, 3): np.array((113, 112, 242)),
  (62, 3): np.array((62, 165, 102)),
  (63, 3): np.array((49, 149, 218)),
  (64, 3): np.array((77, 79, 0)),
  (65, 3): np.array((63, 63, 63)),
  (66, 3): np.array((12, 116, 0)),
  (67, 3): np.array((0, 101, 39)),
  (68, 3): np.array((91, 94, 0)),
  (69, 3): np.array((77, 79, 0)),
  (70, 3): np.array((26, 132, 0)),
  (71, 3): np.array((12, 116, 0)),
  (72, 3): np.array((142, 41, 0)),
  (73, 3): np.array((128, 26, 88)),
  (74, 3): np.array((77, 79, 0)),
  (75, 3): np.array((63, 63, 63)),
  (76, 3): np.array((155, 56, 0)),
  (77, 3): np.array((142, 41, 0)),
  (78, 3): np.array((91, 94, 0)),
  (79, 3): np.array((77, 79, 0)),
  (80, 3): np.array((127, 127, 127)),
  (81, 3): np.array((113, 112, 242)),
  (82, 3): np.array((62, 165, 102)),
  (83, 3): np.array((49, 149, 218)),
  (84, 3): np.array((141, 142, 12)),
  (85, 3): np.array((127, 127, 127)),
  (86, 3): np.array((76, 180, 0)),
  (87, 3): np.array((62, 165, 102)),
  (88, 3): np.array((192, 89, 152)),
  (89, 3): np.array((178, 74, 255)),
  (90, 3): np.array((127, 127, 127)),
  (91, 3): np.array((113, 112, 242)),
  (92, 3): np.array((205, 105, 36)),
  (93, 3): np.array((192, 89, 152)),
  (94, 3): np.array((141, 142, 12)),
  (95, 3): np.array((127, 127, 127)),
  (96, 3): np.array((76, 180, 0)),
  (97, 3): np.array((62, 165, 102)),
  (98, 3): np.array((11, 218, 0)),
  (99, 3): np.array((0, 202, 78)),
  (100, 3): np.array((90, 196, 0)),
  (101, 3): np.array((76, 180, 0)),
  (102, 3): np.array((25, 233, 0)),
  (103, 3): np.array((11, 218, 0)),
  (104, 3): np.array((141, 142, 12)),
  (105, 3): np.array((127, 127, 127)),
  (106, 3): np.array((76, 180, 0)),
  (107, 3): np.array((62, 165, 102)),
  (108, 3): np.array((155, 158, 0)),
  (109, 3): np.array((141, 142, 12)),
  (110, 3): np.array((90, 196, 0)),
  (111, 3): np.array((76, 180, 0)),
  (112, 3): np.array((126, 228, 166)),
  (113, 3): np.array((112, 213, 255)),
  (114, 3): np.array((61, 255, 142)),
  (115, 3): np.array((48, 251, 255)),
  (116, 3): np.array((140, 244, 51)),
  (117, 3): np.array((126, 228, 166)),
  (118, 3): np.array((75, 255, 26)),
  (119, 3): np.array((61, 255, 142)),
  (120, 3): np.array((191, 191, 191)),
  (121, 3): np.array((177, 175, 255)),
  (122, 3): np.array((126, 228, 166)),
  (123, 3): np.array((112, 213, 255)),
  (124, 3): np.array((205, 206, 75)),
  (125, 3): np.array((191, 191, 191)),
  (126, 3): np.array((140, 244, 51)),
  (127, 3): np.array((126, 228, 166)),
  (128, 3): np.array((128, 26, 88)),
  (129, 3): np.array((114, 10, 203)),
  (130, 3): np.array((63, 63, 63)),
  (131, 3): np.array((49, 48, 179)),
  (132, 3): np.array((142, 41, 0)),
  (133, 3): np.array((128, 26, 88)),
  (134, 3): np.array((77, 79, 0)),
  (135, 3): np.array((63, 63, 63)),
  (136, 3): np.array((193, 0, 112)),
  (137, 3): np.array((179, 0, 228)),
  (138, 3): np.array((128, 26, 88)),
  (139, 3): np.array((114, 10, 203)),
  (140, 3): np.array((206, 3, 0)),
  (141, 3): np.array((193, 0, 112)),
  (142, 3): np.array((142, 41, 0)),
  (143, 3): np.array((128, 26, 88)),
  (144, 3): np.array((178, 74, 255)),
  (145, 3): np.array((164, 58, 255)),
  (146, 3): np.array((113, 112, 242)),
  (147, 3): np.array((99, 96, 255)),
  (148, 3): np.array((192, 89, 152)),
  (149, 3): np.array((178, 74, 255)),
  (150, 3): np.array((127, 127, 127)),
  (151, 3): np.array((113, 112, 242)),
  (152, 3): np.array((243, 36, 255)),
  (153, 3): np.array((229, 21, 255)),
  (154, 3): np.array((178, 74, 255)),
  (155, 3): np.array((164, 58, 255)),
  (156, 3): np.array((255, 52, 176)),
  (157, 3): np.array((243, 36, 255)),
  (158, 3): np.array((192, 89, 152)),
  (159, 3): np.array((178, 74, 255)),
  (160, 3): np.array((127, 127, 127)),
  (161, 3): np.array((113, 112, 242)),
  (162, 3): np.array((62, 165, 102)),
  (163, 3): np.array((49, 149, 218)),
  (164, 3): np.array((141, 142, 12)),
  (165, 3): np.array((127, 127, 127)),
  (166, 3): np.array((76, 180, 0)),
  (167, 3): np.array((62, 165, 102)),
  (168, 3): np.array((192, 89, 152)),
  (169, 3): np.array((178, 74, 255)),
  (170, 3): np.array((127, 127, 127)),
  (171, 3): np.array((113, 112, 242)),
  (172, 3): np.array((205, 105, 36)),
  (173, 3): np.array((192, 89, 152)),
  (174, 3): np.array((141, 142, 12)),
  (175, 3): np.array((127, 127, 127)),
  (176, 3): np.array((177, 175, 255)),
  (177, 3): np.array((163, 160, 255)),
  (178, 3): np.array((112, 213, 255)),
  (179, 3): np.array((99, 198, 255)),
  (180, 3): np.array((191, 191, 191)),
  (181, 3): np.array((177, 175, 255)),
  (182, 3): np.array((126, 228, 166)),
  (183, 3): np.array((112, 213, 255)),
  (184, 3): np.array((242, 138, 255)),
  (185, 3): np.array((228, 122, 255)),
  (186, 3): np.array((177, 175, 255)),
  (187, 3): np.array((163, 160, 255)),
  (188, 3): np.array((255, 153, 215)),
  (189, 3): np.array((242, 138, 255)),
  (190, 3): np.array((191, 191, 191)),
  (191, 3): np.array((177, 175, 255)),
  (192, 3): np.array((205, 105, 36)),
  (193, 3): np.array((192, 89, 152)),
  (194, 3): np.array((141, 142, 12)),
  (195, 3): np.array((127, 127, 127)),
  (196, 3): np.array((219, 120, 0)),
  (197, 3): np.array((205, 105, 36)),
  (198, 3): np.array((155, 158, 0)),
  (199, 3): np.array((141, 142, 12)),
  (200, 3): np.array((255, 67, 61)),
  (201, 3): np.array((255, 52, 176)),
  (202, 3): np.array((205, 105, 36)),
  (203, 3): np.array((192, 89, 152)),
  (204, 3): np.array((255, 82, 0)),
  (205, 3): np.array((255, 67, 61)),
  (206, 3): np.array((219, 120, 0)),
  (207, 3): np.array((205, 105, 36)),
  (208, 3): np.array((255, 153, 215)),
  (209, 3): np.array((242, 138, 255)),
  (210, 3): np.array((191, 191, 191)),
  (211, 3): np.array((177, 175, 255)),
  (212, 3): np.array((255, 168, 100)),
  (213, 3): np.array((255, 153, 215)),
  (214, 3): np.array((205, 206, 75)),
  (215, 3): np.array((191, 191, 191)),
  (216, 3): np.array((255, 115, 240)),
  (217, 3): np.array((255, 100, 255)),
  (218, 3): np.array((255, 153, 215)),
  (219, 3): np.array((242, 138, 255)),
  (220, 3): np.array((255, 131, 124)),
  (221, 3): np.array((255, 115, 240)),
  (222, 3): np.array((255, 168, 100)),
  (223, 3): np.array((255, 153, 215)),
  (224, 3): np.array((205, 206, 75)),
  (225, 3): np.array((191, 191, 191)),
  (226, 3): np.array((140, 244, 51)),
  (227, 3): np.array((126, 228, 166)),
  (228, 3): np.array((218, 222, 0)),
  (229, 3): np.array((205, 206, 75)),
  (230, 3): np.array((154, 255, 0)),
  (231, 3): np.array((140, 244, 51)),
  (232, 3): np.array((255, 168, 100)),
  (233, 3): np.array((255, 153, 215)),
  (234, 3): np.array((205, 206, 75)),
  (235, 3): np.array((191, 191, 191)),
  (236, 3): np.array((255, 184, 0)),
  (237, 3): np.array((255, 168, 100)),
  (238, 3): np.array((218, 222, 0)),
  (239, 3): np.array((205, 206, 75)),
  (240, 3): np.array((255, 255, 254)),
  (241, 3): np.array((241, 239, 255)),
  (242, 3): np.array((190, 255, 230)),
  (243, 3): np.array((176, 255, 255)),
  (244, 3): np.array((255, 255, 139)),
  (245, 3): np.array((254, 255, 254)),
  (246, 3): np.array((204, 255, 115)),
  (247, 3): np.array((190, 255, 230)),
  (248, 3): np.array((255, 217, 255)),
  (249, 3): np.array((255, 201, 255)),
  (250, 3): np.array((255, 255, 254)),
  (251, 3): np.array((241, 239, 255)),
  (252, 3): np.array((255, 232, 164)),
  (253, 3): np.array((255, 217, 255)),
  (254, 3): np.array((255, 255, 139)),
  (255, 3): np.array((255, 255, 254)),
}
# 86 unique colours
